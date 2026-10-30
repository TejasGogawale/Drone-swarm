"""MAPPO-style PPO training loop (centralized critic, decentralized actors).

Shared team reward; GAE advantages; PPO-clip. Works on batched RescueEnv instances.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .model import ActorCritic


def obs_to_tensors(obs, device):
    type_ids = torch.tensor([0 if t == "drone" else 1 for t in obs["types"]],
                            dtype=torch.long).to(device)
    return (
        torch.from_numpy(obs["agent_feats"]).to(device),
        torch.from_numpy(obs["goal_feats"]).to(device),
        torch.from_numpy(obs["goal_valid"]).to(device),
        type_ids,
        torch.from_numpy(obs["comm_adj"]).to(device),
        torch.from_numpy(obs["coverage"]).to(device),
    )


def forward_one(model, obs, device):
    af, gf, gv, type_ids, ca, cov = obs_to_tensors(obs, device)
    return model(af, gf, gv, type_ids, ca, cov)


@torch.no_grad()
def act(model, obs, device, deterministic=False):
    logits, value = forward_one(model, obs, device)
    gv = torch.from_numpy(obs["goal_valid"]).to(device)
    # agents with no valid goal: fall back to 'stay' (slot 0)
    no_valid = (~gv).all(dim=1)
    l = logits.clone()
    l = l.masked_fill(no_valid.unsqueeze(1), 0.0)
    probs = torch.softmax(l, dim=-1)
    dist = Categorical(probs)
    action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    logprob = dist.log_prob(action)
    return action, logprob, value


def compute_gae(rewards, values, dones, gamma, lam):
    """rewards/values/dones: lists over T, each element shape [B].
    Returns advantages and returns as [T, B] arrays (GAE per env)."""
    r = np.asarray(rewards, dtype=np.float32)    # [T, B]
    v = np.asarray(values, dtype=np.float32)
    d = np.asarray(dones, dtype=np.float32)      # 1.0 if done
    T, B = r.shape
    adv = np.zeros_like(r)
    last_adv = np.zeros(B, dtype=np.float32)
    next_val = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - d[t]                     # [B]
        delta = r[t] + gamma * next_val * nonterm - v[t]
        last_adv = delta + gamma * lam * nonterm * last_adv
        adv[t] = last_adv
        next_val = v[t]
    returns = adv + v
    return adv, returns


class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 vf_coef=0.5, ent_coef=0.01, epochs=10, minibatch=None,
                 max_grad_norm=0.5, device="cpu"):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, obs_list, actions, logprobs, values, rewards, dones):
        """obs_list: list over T of list over B of obs dicts.
        actions/logprobs: [T][B][nA]; values/rewards/dones: [T][B]."""
        T = len(obs_list)
        nA = actions[0].shape[1]
        adv, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)

        # flatten: each obs dict covers one env (nA agents)
        flat_obs = [obs for t in range(T) for obs in obs_list[t]]          # length T*B
        flat_actions = np.concatenate([actions[t].reshape(-1) for t in range(T)])       # T*B*nA
        flat_logprobs = np.concatenate([logprobs[t].reshape(-1) for t in range(T)])
        flat_adv = np.concatenate([np.tile(adv[t][:, None], (1, nA)).reshape(-1) for t in range(T)])
        flat_returns = np.concatenate([np.tile(returns[t][:, None], (1, nA)).reshape(-1) for t in range(T)])

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        n_env = len(flat_obs)
        env_idx = np.arange(n_env)
        mb = self.minibatch or n_env
        n_batches = max(1, n_env // mb)

        for _ in range(self.epochs):
            np.random.shuffle(env_idx)
            for bi in range(n_batches):
                batch_env = env_idx[bi * mb:(bi + 1) * mb]
                if len(batch_env) == 0:
                    continue
                self._update_step(batch_env, flat_obs, flat_actions, flat_logprobs,
                                  flat_adv, flat_returns, nA)

    def _update_step(self, batch_env, flat_obs, flat_actions, flat_logprobs,
                     flat_adv, flat_returns, nA):
        device = self.device
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        n = 0
        for e in batch_env:
            obs = flat_obs[e]
            logits, value = forward_one(self.model, obs, device)
            gv = torch.from_numpy(obs["goal_valid"]).to(device)
            no_valid = (~gv).all(dim=1)
            l = logits.clone().masked_fill(no_valid.unsqueeze(1), 0.0)
            dist = Categorical(torch.softmax(l, dim=-1))
            a = torch.from_numpy(flat_actions[e * nA:(e + 1) * nA]).long().to(device)
            new_logprob = dist.log_prob(a)
            old_logprob = torch.from_numpy(flat_logprobs[e * nA:(e + 1) * nA]).to(device)
            adv_t = torch.from_numpy(flat_adv[e * nA:(e + 1) * nA]).to(device)
            ret_t = torch.from_numpy(flat_returns[e * nA:(e + 1) * nA]).to(device)

            ratio = torch.exp(new_logprob - old_logprob)
            pg1 = ratio * adv_t
            pg2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            policy_loss += -torch.min(pg1, pg2).mean()
            value_loss += F.mse_loss(value.squeeze(-1).expand_as(ret_t), ret_t)
            entropy += dist.entropy().mean()
            n += 1

        policy_loss = policy_loss / max(1, n)
        value_loss = value_loss / max(1, n)
        entropy = entropy / max(1, n)
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
