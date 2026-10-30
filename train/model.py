"""Actor-critic network with a GraphSAGE-style GNN (pure PyTorch).

Graph nodes: agents (encoded per robot type) + goal candidates (frontiers).
Edges: agent<->agent within comms range, and agent<->its own goal candidates.
Message passing lets each agent reason about neighbors and map nodes -> the GNN
for inter-robot communication + collaborative map, as required.

Decentralized actors (per-type heads), centralized critic (global pooling + coverage).
Pure-PyTorch implementation for clean TorchScript export (no torch_geometric at
inference time).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-aggregate `src` into `dim_size` rows indexed by `index` (pure torch)."""
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    cnt.index_add_(0, index, torch.ones_like(src[:, :1]))
    return out / cnt.clamp_min(1.0)


class SAGELayer(nn.Module):
    """GraphSAGE mean-aggregator layer (self + neighbor mean)."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d)
        self.lin_neigh = nn.Linear(in_d, out_d)

    def forward(self, x, edge_index):
        # edge_index: [2, E], message flows src -> dst
        src, dst = edge_index
        neigh = _scatter_mean(x[src], dst, x.shape[0])
        return F.relu(self.lin(x) + self.lin_neigh(neigh))


class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_d))

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, fa, fg, n_goals=20, hidden=64,
                 n_layers=2, use_gnn=True):
        super().__init__()
        self.hidden = hidden
        self.n_goals = n_goals
        self.use_gnn = use_gnn
        # per-type agent encoders
        self.drone_enc = MLP(fa, hidden)
        self.rover_enc = MLP(fa, hidden)
        self.goal_enc = MLP(fg, hidden)
        self.convs = nn.ModuleList([SAGELayer(hidden, hidden) for _ in range(n_layers)]) if use_gnn else None
        # actor heads per type: score(agent_emb, goal_emb) -> scalar
        self.scorer = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))
        # critic: global mean agent embedding + coverage -> value
        self.critic = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def _encode_agents(self, agent_feats, type_ids):
        # type_ids: [nA] long, 0=drone, 1=rover. Vectorized (TorchScript-friendly).
        mask = (type_ids == 0).float().unsqueeze(-1)   # [nA, 1]
        h = self.drone_enc(agent_feats) * mask + self.rover_enc(agent_feats) * (1 - mask)
        return h                                        # [nA, hidden]

    def _build_edge_index(self, nA, comm_adj, device):
        """Build directed edge list for agents (0..nA-1) and goals (nA..nA+nA*NG-1)."""
        rows, cols = [], []
        # agent-agent (bidirectional)
        idx = torch.nonzero(comm_adj > 0.5, as_tuple=False)
        for i, j in idx.tolist():
            rows += [i, j]
            cols += [j, i]
        # agent-goal (bidirectional): agent i <-> its NG goal nodes
        NG = self.n_goals
        for i in range(nA):
            base = nA + i * NG
            for k in range(NG):
                g = base + k
                rows += [i, g]
                cols += [g, i]
        if not rows:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)

    def forward(self, agent_feats, goal_feats, goal_valid, type_ids, comm_adj, coverage):
        """Returns (logits [nA, NG] masked, value [1])."""
        nA = agent_feats.shape[0]
        NG = self.n_goals
        device = agent_feats.device
        hA = self._encode_agents(agent_feats, type_ids)              # [nA, hidden]
        hG = self.goal_enc(goal_feats.reshape(-1, goal_feats.shape[-1]))  # [nA*NG, hidden]

        if self.use_gnn:
            x = torch.cat([hA, hG], dim=0)                      # [nA + nA*NG, hidden]
            ei = self._build_edge_index(nA, comm_adj, device)
            for conv in self.convs:
                x = conv(x, ei)
            hA = x[:nA]
            hG = x[nA:]
        hG = hG.reshape(nA, NG, self.hidden)

        # actor logits
        logits = []
        for i in range(nA):
            a = hA[i].unsqueeze(0).expand(NG, self.hidden)      # [NG, hidden]
            inp = torch.cat([a, hG[i]], dim=-1)                 # [NG, 2h]
            logits.append(self.scorer(inp).squeeze(-1))         # [NG]
        logits = torch.stack(logits, dim=0)                     # [nA, NG]
        logits = logits.masked_fill(~goal_valid, float("-inf"))

        # critic value (centralized)
        g = hA.mean(dim=0)                                      # [hidden]
        value = self.critic(torch.cat([g, coverage], dim=-1))   # [1]
        return logits, value
