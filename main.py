import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # to work on HPC
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Models
# ===========================

class RiskEncoder(nn.Module):
    """
    GRU-based encoder for latent risk context Z_t.
    Input x_t = [state, prev_cost].
    """
    def __init__(self, state_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = state_dim + 1  # state + cost
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.z_proj = nn.Linear(hidden_dim, latent_dim)

    def forward_sequence(self, states, costs):
        """
        states: (T, state_dim)
        costs:  (T,) or (T,1)
        Returns:
            Z: (T, latent_dim)   (detached for PG later)
        NOTE: here we mainly use online step-wise GRU; this is for completeness.
        """
        T = states.shape[0]
        x = torch.cat([states, costs.view(T, 1)], dim=-1).unsqueeze(0)  # (1,T,input_dim)
        h0 = torch.zeros(1, 1, self.gru.hidden_size, device=states.device)
        h_seq, _ = self.gru(x, h0)  # (1,T,hidden_dim)
        h_seq = h_seq.squeeze(0)
        z = self.z_proj(h_seq)
        return z

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
    We approximate b(s,z,eta) ≈ E[(C - eta)_+ | s,z] with a simple MLP.
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


# ===========================
# Rollout
# ===========================

class Transition:
    __slots__ = ("states", "latents", "actions", "returns", "traj_cost")

    def __init__(self, states, latents, actions, returns, traj_cost):
        self.states = states      # (T, state_dim)
        self.latents = latents    # (T, latent_dim)
        self.actions = actions    # (T,)
        self.returns = returns    # (T,) reward returns
        self.traj_cost = traj_cost  # scalar (float)


def rollout_batch(env, encoder, policy, N, T_max, cost_fn, gamma=0.99):
    """
    Collect N on-policy trajectories with encoder stop-gradient.
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

        t = 0
        while not done and t < T_max:
            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,state_dim)

            # Encoder step (full stop-grad for actor)
            with torch.no_grad():
                h_next, z_t = encoder.step(h_t, s_t, prev_cost_t)
                h_next = h_next.detach()
                z_t = z_t.detach()  # (1,latent_dim)
                dist = policy(s_t, z_t)
                a_t = dist.sample()

            a_int = int(a_t.item())
            next_obs, r_t, terminated, truncated, _ = env.step(a_int)
            done = terminated or truncated

            c_t = cost_fn(obs, a_int, r_t, next_obs, done)

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

        # Discounted reward returns
        G_R = torch.zeros(T, device=device)
        G = 0.0
        for t_rev in reversed(range(T)):
            G = rewards[t_rev] + gamma * G
            G_R[t_rev] = G

        # Discounted trajectory cost
        gammas = (gamma ** torch.arange(T, device=device, dtype=torch.float32))
        C_total = torch.sum(gammas * costs_t).item()

        transitions.append(Transition(states, latents, actions, G_R, C_total))

    return transitions


# ===========================
# Updates
# ===========================

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
    Target is trajectory-level tail loss broadcast over time.
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
            b = tail_net(states, latents, torch.tensor(eta, device=device))  # (T,)
            C_eta_plus = max(C_total - eta, 0.0)
            cvar_term = C_eta_plus - b  # broadcast baseline

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

    # Estimate CVaR
    cvar_est = estimate_cvar(costs, alpha_cvar)

    # Dual update
    lambda_dual = lambda_dual + alpha_lambda * (cvar_est - beta_constraint)
    lambda_dual = max(0.0, min(lambda_dual, 10.0))

    # RU threshold update
    indicator = (C >= eta).float()
    subgrad = 1.0 - (indicator.mean().item() / (1.0 - alpha_cvar))
    eta = eta - alpha_eta * lambda_dual * subgrad

    return lambda_dual, eta, cvar_est


# ===========================
# Training loops
# ===========================

def train_lc_cvar_pg(env_name="CartPole-v1",
                     num_iterations=500,
                     N=16,
                     T_max=200,
                     gamma=0.99,
                     alpha_cvar=0.9,
                     beta_constraint=10.0):

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = 16

    def cost_fn(obs, action, reward, next_obs, done):
        # Example: cost = |pole angle| + big cost at failure
        angle = next_obs[2]
        c = abs(angle)
        if done and reward < 1.0:
            c += 5.0
        return c

    encoder = RiskEncoder(state_dim, latent_dim).to(device)
    policy = PolicyNet(state_dim, latent_dim, action_dim).to(device)
    value_net = ValueNet(state_dim, latent_dim).to(device)
    tail_net = TailBaselineNet(state_dim, latent_dim).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optim_value = optim.Adam(value_net.parameters(), lr=1e-3)
    optim_tail = optim.Adam(tail_net.parameters(), lr=1e-3)

    lambda_dual = 0.0
    eta = 0.0

    alpha_lambda = 1e-3
    alpha_eta = 5e-4

    returns_history = []
    cvar_history = []
    lambda_history = []

    for it in range(num_iterations):
        transitions = rollout_batch(env, encoder, policy, N, T_max, cost_fn, gamma)
        traj_costs = [tr.traj_cost for tr in transitions]
        traj_returns = [tr.returns[0].item() for tr in transitions]

        v_loss = update_value_net(value_net, optim_value, transitions)
        tail_loss = update_tail_baseline(tail_net, optim_tail, transitions, eta)
        pi_loss = actor_update_lc_cvar(policy, value_net, tail_net,
                                       optim_policy, transitions,
                                       lambda_dual, eta, alpha_cvar)

        lambda_dual, eta, cvar_est = dual_eta_update(
            traj_costs, lambda_dual, eta,
            alpha_cvar, alpha_lambda, alpha_eta, beta_constraint
        )

        returns_history.append(np.mean(traj_returns))
        cvar_history.append(cvar_est)
        lambda_history.append(lambda_dual)

        if (it + 1) % 10 == 0:
            print(f"[LC-CVaR-PG Iter {it+1}] "
                  f"Return: {np.mean(traj_returns):.2f}  "
                  f"CVaR: {cvar_est:.2f}  "
                  f"lambda: {lambda_dual:.3f}  "
                  f"V_loss: {v_loss:.3f}  Tail_loss: {tail_loss:.3f}  "
                  f"Pi_loss: {pi_loss:.3f}")

    env.close()
    return returns_history, cvar_history, lambda_history


def train_vanilla_pg(env_name="CartPole-v1",
                     num_iterations=500,
                     N=16,
                     T_max=200,
                     gamma=0.99):

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    latent_dim = 16

    def cost_fn(obs, action, reward, next_obs, done):
        # cost irrelevant for vanilla PG
        return 0.0

    encoder = RiskEncoder(state_dim, latent_dim).to(device)
    policy = PolicyNet(state_dim, latent_dim, action_dim).to(device)
    value_net = ValueNet(state_dim, latent_dim).to(device)

    optim_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optim_value = optim.Adam(value_net.parameters(), lr=1e-3)

    returns_history = []

    for it in range(num_iterations):
        transitions = rollout_batch(env, encoder, policy, N, T_max, cost_fn, gamma)
        traj_returns = [tr.returns[0].item() for tr in transitions]

        v_loss = update_value_net(value_net, optim_value, transitions)
        pi_loss = actor_update_vanilla_pg(policy, value_net, optim_policy, transitions)

        returns_history.append(np.mean(traj_returns))

        if (it + 1) % 10 == 0:
            print(f"[Vanilla PG Iter {it+1}] "
                  f"Return: {np.mean(traj_returns):.2f}  "
                  f"V_loss: {v_loss:.3f}  Pi_loss: {pi_loss:.3f}")

    env.close()
    return returns_history


if __name__ == "__main__":
    # Example local run + plotting for comparison
    iters = 300
    ret_vanilla = train_vanilla_pg(num_iterations=iters)
    ret_lc, cvar_lc, lam_lc = train_lc_cvar_pg(num_iterations=iters,
                                               beta_constraint=10.0)

    plt.figure()
    plt.plot(ret_vanilla, label="Vanilla PG")
    plt.plot(ret_lc, label="LC-CVaR-PG")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Average return")
    plt.savefig("returns_comparison.png")

    plt.figure()
    plt.plot(cvar_lc, label="LC-CVaR-PG CVaR")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("CVaR (est.)")
    plt.savefig("cvar_lc_cvar_pg.png")