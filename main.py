import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # safe on HPC / headless
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# Models: encoder, policy, value, tail baseline
# ======================================

class RiskEncoder(nn.Module):
    """
    GRU-based encoder for latent risk context Z_t.
    Input x_t = [state, prev_cost].
    """
    def __init__(self, state_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = state_dim + 1
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.z_proj = nn.Linear(hidden_dim, latent_dim)

    def init_hidden(self):
        return torch.zeros(1, 1, self.gru.hidden_size, device=device)

    def step(self, h_t, state_t, prev_cost_t):
        """
        One GRU step.
        h_t: (1,1,H)
        state_t: (1,state_dim)
        prev_cost_t: scalar tensor
        """
        x_t = torch.cat([state_t, prev_cost_t.view(1, 1)], dim=-1).unsqueeze(0)  # (1,1,input_dim)
        h_next, _ = self.gru(x_t, h_t)
        z_t = self.z_proj(h_next.squeeze(0))  # (1, latent_dim)
        return h_next, z_t


class PolicyNet(nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


class ValueNet(nn.Module):
    """V_omega(s, z) for reward baseline."""
    def __init__(self, state_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, z):
        return self.net(torch.cat([s, z], dim=-1)).squeeze(-1)


class TailBaselineNet(nn.Module):
    """
    b_psi(s, z, eta): latent-conditioned tail baseline.
    Approximates E[(C - eta)_+ | s,z].
    """
    def __init__(self, state_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, z, eta):
        T = s.shape[0]
        eta_vec = torch.full((T, 1), eta, device=s.device)
        x = torch.cat([s, z, eta_vec], dim=-1)
        return self.net(x).squeeze(-1)


# ======================================
# Rollout structure
# ======================================

class Transition:
    __slots__ = ("states", "latents", "actions",
                 "returns", "traj_cost", "crashed")

    def __init__(self, states, latents, actions,
                 returns, traj_cost, crashed):
        self.states = states      # (T, state_dim)
        self.latents = latents    # (T, latent_dim)
        self.actions = actions    # (T,)
        self.returns = returns    # (T,) reward returns
        self.traj_cost = traj_cost  # scalar
        self.crashed = crashed      # bool


def rollout_batch(env, encoder, policy, N, T_max, cost_fn, gamma=0.99):
    """
    Collect N on-policy trajectories for LunarLander
    with encoder stop-gradient, and cost_fn defining risk.
    """
    transitions = []

    encoder.eval()
    policy.eval()

    for _ in range(N):
        obs, _ = env.reset()
        done = False

        h_t = encoder.init_hidden()
        prev_cost_t = torch.tensor(0.0, device=device)

        states = []
        latents = []
        actions = []
        rewards = []
        costs = []

        crashed = False
        t = 0
        while not done and t < T_max:
            s_t = torch.tensor(obs, dtype=torch.float32,
                               device=device).unsqueeze(0)  # (1,state_dim)

            # Encoder step, full stop-gradient for actor
            with torch.no_grad():
                h_next, z_t = encoder.step(h_t, s_t, prev_cost_t)
                h_next = h_next.detach()
                z_t = z_t.detach()
                dist = policy(s_t, z_t)
                a_t = dist.sample()

            a_int = int(a_t.item())
            next_obs, r_t, terminated, truncated, info = env.step(a_int)
            done = terminated or truncated

            c_t, crash_flag = cost_fn(obs, a_int, r_t, next_obs, done)
            crashed = crashed or crash_flag

            states.append(s_t.squeeze(0))
            latents.append(z_t.squeeze(0))
            actions.append(a_int)
            rewards.append(r_t)
            costs.append(c_t)

            obs = next_obs
            h_t = h_next
            prev_cost_t = torch.tensor(c_t, dtype=torch.float32, device=device)
            t += 1

        T = len(states)
        states = torch.stack(states, dim=0)
        latents = torch.stack(latents, dim=0)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        costs_t = torch.tensor(costs, device=device, dtype=torch.float32)

        # Discounted reward
        G_R = torch.zeros(T, device=device)
        G = 0.0
        for t_rev in reversed(range(T)):
            G = rewards[t_rev] + gamma * G
            G_R[t_rev] = G

        # Discounted trajectory cost
        gammas = (gamma ** torch.arange(T, device=device, dtype=torch.float32))
        C_total = torch.sum(gammas * costs_t).item()

        transitions.append(Transition(states, latents, actions,
                                      G_R, C_total, crashed))

    return transitions


# ======================================
# Update utilities
# ======================================

def update_value_net(value_net, optimizer, transitions):
    value_net.train()
    states = torch.cat([tr.states for tr in transitions], dim=0)
    latents = torch.cat([tr.latents for tr in transitions], dim=0)
    returns = torch.cat([tr.returns for tr in transitions], dim=0)

    V = value_net(states, latents)
    loss = ((V - returns) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def update_tail_baseline(tail_net, optimizer, transitions, eta):
    """
    Train b_psi(s,z,eta) ≈ (C - eta)_+ as function of (s,z).
    """
    tail_net.train()
    all_states = []
    all_latents = []
    all_targets = []

    for tr in transitions:
        T = tr.states.shape[0]
        states = tr.states
        latents = tr.latents
        C_total = tr.traj_cost
        target = max(C_total - eta, 0.0)
        targets = torch.full((T,), target, device=device)
        all_states.append(states)
        all_latents.append(latents)
        all_targets.append(targets)

    states = torch.cat(all_states, dim=0)
    latents = torch.cat(all_latents, dim=0)
    targets = torch.cat(all_targets, dim=0)

    preds = tail_net(states, latents, torch.tensor(eta, device=device))
    loss = ((preds - targets) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def actor_update_lc_cvar(policy, value_net, tail_net,
                         optimizer, transitions,
                         lambda_dual, eta, alpha_cvar):
    """
    LC–CVaR–PG actor update with tail baseline.
    """
    policy.train()
    value_net.eval()
    tail_net.eval()

    all_logp = []
    all_advR = []
    all_cvar_term = []

    for tr in transitions:
        states = tr.states.detach()
        latents = tr.latents.detach()
        actions = tr.actions
        returns = tr.returns
        C_total = tr.traj_cost

        with torch.no_grad():
            V = value_net(states, latents)
            A_R = returns - V
            b = tail_net(states, latents, torch.tensor(eta, device=device))
            C_eta_plus = max(C_total - eta, 0.0)
            cvar_term = C_eta_plus - b

        dist = policy(states, latents)
        logp = dist.log_prob(actions)

        all_logp.append(logp)
        all_advR.append(A_R)
        all_cvar_term.append(cvar_term)

    logp_all = torch.cat(all_logp, dim=0)
    A_R_all = torch.cat(all_advR, dim=0)
    cvar_all = torch.cat(all_cvar_term, dim=0)

    loss = - (logp_all * (A_R_all +
                          (lambda_dual / (1.0 - alpha_cvar)) * cvar_all)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def actor_update_cvar_no_baseline(policy, value_net,
                                  optimizer, transitions,
                                  lambda_dual, eta, alpha_cvar):
    """
    CVaR-PG actor update WITHOUT tail baseline (high variance).
    """
    policy.train()
    value_net.eval()

    all_logp = []
    all_advR = []
    all_cvar_term = []

    for tr in transitions:
        states = tr.states.detach()
        latents = tr.latents.detach()
        actions = tr.actions
        returns = tr.returns
        C_total = tr.traj_cost

        with torch.no_grad():
            V = value_net(states, latents)
            A_R = returns - V
            C_eta_plus = max(C_total - eta, 0.0)
            cvar_term = torch.full((states.shape[0],), C_eta_plus, device=device)

        dist = policy(states, latents)
        logp = dist.log_prob(actions)

        all_logp.append(logp)
        all_advR.append(A_R)
        all_cvar_term.append(cvar_term)

    logp_all = torch.cat(all_logp, dim=0)
    A_R_all = torch.cat(all_advR, dim=0)
    cvar_all = torch.cat(all_cvar_term, dim=0)

    loss = - (logp_all * (A_R_all +
                          (lambda_dual / (1.0 - alpha_cvar)) * cvar_all)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def actor_update_vanilla_pg(policy, value_net, optimizer, transitions):
    """
    Risk-neutral PG baseline.
    """
    policy.train()
    value_net.eval()

    all_logp = []
    all_advR = []

    for tr in transitions:
        states = tr.states.detach()
        latents = tr.latents.detach()
        actions = tr.actions
        returns = tr.returns

        with torch.no_grad():
            V = value_net(states, latents)
            A_R = returns - V

        dist = policy(states, latents)
        logp = dist.log_prob(actions)

        all_logp.append(logp)
        all_advR.append(A_R)

    logp_all = torch.cat(all_logp, dim=0)
    A_R_all = torch.cat(all_advR, dim=0)

    loss = - (logp_all * A_R_all).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def estimate_cvar(costs, alpha_cvar):
    C = torch.tensor(costs, device=device)
    q = torch.quantile(C, alpha_cvar)
    tail = C[C >= q]
    if tail.numel() == 0:
        return 0.0
    return float(tail.mean().item())


def dual_eta_update(costs, lambda_dual, eta, alpha_cvar,
                    alpha_lambda, alpha_eta, beta_constraint):
    C = torch.tensor(costs, device=device)
    N = C.shape[0]

    cvar_est = estimate_cvar(costs, alpha_cvar)

    # Dual update
    lambda_dual = lambda_dual + alpha_lambda * (cvar_est - beta_constraint)
    lambda_dual = max(0.0, min(lambda_dual, 50.0))

    # RU threshold update
    indicator = (C >= eta).float()
    subgrad = 1.0 - (indicator.mean().item() / (1.0 - alpha_cvar))
    eta = eta - alpha_eta * lambda_dual * subgrad

    return lambda_dual, eta, cvar_est


# ======================================
# LunarLander risk cost
# ======================================

def lunar_cost_fn(obs, action, reward, next_obs, done):
    """
    Design a risk-sensitive cost for LunarLander.
    Heavily penalize crashes; penalize unstable attitude and high speed.
    """
    # next_obs: [x, y, vx, vy, theta, vtheta, leg1, leg2]
    x, y, vx, vy, theta, vtheta, leg1, leg2 = next_obs

    cost = 0.0
    # penalize large velocities and angle (risky behavior)
    cost += 0.5 * (abs(vx) + abs(vy))
    cost += 0.3 * abs(theta) + 0.1 * abs(vtheta)

    crashed = False
    if done and reward < 0:  # crash episode in default env
        cost += 100.0
        crashed = True

    return cost, crashed


# ======================================
# Training loops for three algorithms
# ======================================

def train_lc_cvar_pg(num_iterations=400,
                     N=16,
                     T_max=500,
                     gamma=0.99,
                     alpha_cvar=0.9,
                     beta_constraint=60.0):

    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = 16

    encoder = RiskEncoder(state_dim, latent_dim).to(device)
    policy = PolicyNet(state_dim, latent_dim, action_dim).to(device)
    value_net = ValueNet(state_dim, latent_dim).to(device)
    tail_net = TailBaselineNet(state_dim, latent_dim).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optim_value = optim.Adam(value_net.parameters(), lr=1e-3)
    optim_tail = optim.Adam(tail_net.parameters(), lr=1e-3)

    lambda_dual = 0.0
    eta = 0.0
    alpha_lambda = 5e-4
    alpha_eta = 2e-4

    ret_hist, cvar_hist, crash_hist = [], [], []

    for it in range(num_iterations):
        transitions = rollout_batch(env, encoder, policy, N, T_max,
                                    lunar_cost_fn, gamma)
        traj_costs = [tr.traj_cost for tr in transitions]
        traj_returns = [tr.returns[0].item() for tr in transitions]
        traj_crashes = [1.0 if tr.crashed else 0.0 for tr in transitions]

        v_loss = update_value_net(value_net, optim_value, transitions)
        tail_loss = update_tail_baseline(tail_net, optim_tail, transitions, eta)
        pi_loss = actor_update_lc_cvar(policy, value_net, tail_net,
                                       optim_policy, transitions,
                                       lambda_dual, eta, alpha_cvar)

        lambda_dual, eta, cvar_est = dual_eta_update(
            traj_costs, lambda_dual, eta,
            alpha_cvar, alpha_lambda, alpha_eta, beta_constraint
        )

        ret_hist.append(np.mean(traj_returns))
        cvar_hist.append(cvar_est)
        crash_hist.append(np.mean(traj_crashes))

        if (it + 1) % 10 == 0:
            print(f"[LC-CVaR Iter {it+1}] "
                  f"Ret {np.mean(traj_returns):6.2f}  "
                  f"CVaR {cvar_est:7.2f}  "
                  f"Crash {np.mean(traj_crashes):4.2f}  "
                  f"lambda {lambda_dual:6.3f}  "
                  f"V {v_loss:.3f}  Tail {tail_loss:.3f}  Pi {pi_loss:.3f}")

    env.close()
    return ret_hist, cvar_hist, crash_hist


def train_cvar_no_baseline(num_iterations=400,
                           N=16,
                           T_max=500,
                           gamma=0.99,
                           alpha_cvar=0.9,
                           beta_constraint=60.0):

    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = 16

    encoder = RiskEncoder(state_dim, latent_dim).to(device)
    policy = PolicyNet(state_dim, latent_dim, action_dim).to(device)
    value_net = ValueNet(state_dim, latent_dim).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optim_value = optim.Adam(value_net.parameters(), lr=1e-3)

    lambda_dual = 0.0
    eta = 0.0
    alpha_lambda = 5e-4
    alpha_eta = 2e-4

    ret_hist, cvar_hist, crash_hist = [], [], []

    for it in range(num_iterations):
        transitions = rollout_batch(env, encoder, policy, N, T_max,
                                    lunar_cost_fn, gamma)
        traj_costs = [tr.traj_cost for tr in transitions]
        traj_returns = [tr.returns[0].item() for tr in transitions]
        traj_crashes = [1.0 if tr.crashed else 0.0 for tr in transitions]

        v_loss = update_value_net(value_net, optim_value, transitions)
        pi_loss = actor_update_cvar_no_baseline(
            policy, value_net, optim_policy,
            transitions, lambda_dual, eta, alpha_cvar
        )

        lambda_dual, eta, cvar_est = dual_eta_update(
            traj_costs, lambda_dual, eta,
            alpha_cvar, alpha_lambda, alpha_eta, beta_constraint
        )

        ret_hist.append(np.mean(traj_returns))
        cvar_hist.append(cvar_est)
        crash_hist.append(np.mean(traj_crashes))

        if (it + 1) % 10 == 0:
            print(f"[CVaR-noBL Iter {it+1}] "
                  f"Ret {np.mean(traj_returns):6.2f}  "
                  f"CVaR {cvar_est:7.2f}  "
                  f"Crash {np.mean(traj_crashes):4.2f}  "
                  f"lambda {lambda_dual:6.3f}  "
                  f"V {v_loss:.3f}  Pi {pi_loss:.3f}")

    env.close()
    return ret_hist, cvar_hist, crash_hist


def train_vanilla_pg(num_iterations=400,
                     N=16,
                     T_max=500,
                     gamma=0.99):

    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = 16

    encoder = RiskEncoder(state_dim, latent_dim).to(device)
    policy = PolicyNet(state_dim, latent_dim, action_dim).to(device)
    value_net = ValueNet(state_dim, latent_dim).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optim_value = optim.Adam(value_net.parameters(), lr=1e-3)

    ret_hist, cvar_hist, crash_hist = [], [], []

    for it in range(num_iterations):
        transitions = rollout_batch(env, encoder, policy, N, T_max,
                                    lunar_cost_fn, gamma)
        traj_returns = [tr.returns[0].item() for tr in transitions]
        traj_costs = [tr.traj_cost for tr in transitions]
        traj_crashes = [1.0 if tr.crashed else 0.0 for tr in transitions]

        v_loss = update_value_net(value_net, optim_value, transitions)
        pi_loss = actor_update_vanilla_pg(policy, value_net, optim_policy, transitions)

        cvar_est = estimate_cvar(traj_costs, alpha_cvar=0.9)  # fixed alpha for logging

        ret_hist.append(np.mean(traj_returns))
        cvar_hist.append(cvar_est)
        crash_hist.append(np.mean(traj_crashes))

        if (it + 1) % 10 == 0:
            print(f"[Vanilla Iter {it+1}] "
                  f"Ret {np.mean(traj_returns):6.2f}  "
                  f"CVaR {cvar_est:7.2f}  "
                  f"Crash {np.mean(traj_crashes):4.2f}  "
                  f"V {v_loss:.3f}  Pi {pi_loss:.3f}")

    env.close()
    return ret_hist, cvar_hist, crash_hist


# ======================================
# Main experiment
# ======================================

if __name__ == "__main__":
    num_iters = 400

    print("=== Training Vanilla PG ===")
    ret_van, cvar_van, crash_van = train_vanilla_pg(num_iterations=num_iters)

    print("=== Training CVaR-PG (no baseline) ===")
    ret_cvar_nb, cvar_cvar_nb, crash_cvar_nb = train_cvar_no_baseline(
        num_iterations=num_iters, beta_constraint=60.0)

    print("=== Training LC–CVaR–PG (ours) ===")
    ret_lc, cvar_lc, crash_lc = train_lc_cvar_pg(
        num_iterations=num_iters, beta_constraint=60.0)

    # ---- Plots ----
    iters = np.arange(num_iters)

    plt.figure()
    plt.plot(iters, ret_van, label="Vanilla PG")
    plt.plot(iters, ret_cvar_nb, label="CVaR-PG (no baseline)")
    plt.plot(iters, ret_lc, label="LC–CVaR–PG (ours)")
    plt.xlabel("Iteration")
    plt.ylabel("Average return")
    plt.legend()
    plt.title("LunarLander: Return")
    plt.savefig("ll_return_compare.png", dpi=150)

    plt.figure()
    plt.plot(iters, cvar_van, label="Vanilla PG")
    plt.plot(iters, cvar_cvar_nb, label="CVaR-PG (no baseline)")
    plt.plot(iters, cvar_lc, label="LC–CVaR–PG (ours)")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated CVaR (alpha=0.9)")
    plt.legend()
    plt.title("LunarLander: CVaR of cost")
    plt.savefig("ll_cvar_compare.png", dpi=150)

    plt.figure()
    plt.plot(iters, crash_van, label="Vanilla PG")
    plt.plot(iters, crash_cvar_nb, label="CVaR-PG (no baseline)")
    plt.plot(iters, crash_lc, label="LC–CVaR–PG (ours)")
    plt.xlabel("Iteration")
    plt.ylabel("Crash rate")
    plt.legend()
    plt.title("LunarLander: Crash rate")
    plt.savefig("ll_crash_compare.png", dpi=150)