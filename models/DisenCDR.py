import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

from utils import full_rank_eval


# ───────────────────────── Graph primitives ─────────────────────────

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        return self.leakyrelu(self.gc1(x, adj))


# ───────────────────── single-domain VBGE ───────────────────────

class _SingleDGCNLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        fd, hd, dr, lk = opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"]
        self.gc1 = GCN(fd, hd, dr, lk)
        self.gc2 = GCN(fd, hd, dr, lk)
        self.gc3 = GCN(hd, fd, dr, lk)
        self.gc4 = GCN(hd, fd, dr, lk)
        self.user_union = nn.Linear(fd * 2, fd)
        self.item_union = nn.Linear(fd * 2, fd)

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        Item_ho = self.gc4(self.gc2(vfea, UV_adj), VU_adj)
        User = self.user_union(torch.cat((User_ho, ufea), dim=1))
        Item = self.item_union(torch.cat((Item_ho, vfea), dim=1))
        return F.relu(User), F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        User = self.user_union(torch.cat((User_ho, ufea), dim=1))
        return F.relu(User)


class _SingleLastLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        fd, hd, dr, lk = opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"]
        self.gc1 = GCN(fd, hd, dr, lk)
        self.gc2 = GCN(fd, hd, dr, lk)
        self.gc3_mean = GCN(hd, fd, dr, lk)
        self.gc3_logstd = GCN(hd, fd, dr, lk)
        self.gc4_mean = GCN(hd, fd, dr, lk)
        self.gc4_logstd = GCN(hd, fd, dr, lk)
        self.user_union_mean = nn.Linear(fd * 2, fd)
        self.user_union_logstd = nn.Linear(fd * 2, fd)
        self.item_union_mean = nn.Linear(fd * 2, fd)
        self.item_union_logstd = nn.Linear(fd * 2, fd)
        self.kld_loss = 0

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        sigma_1 = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logsigma_1)), max=10)
        sigma_2 = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logsigma_2)), max=10)
        return kl_divergence(Normal(mu_1, sigma_1), Normal(mu_2, sigma_2)).mean(dim=0).sum()

    def reparameters(self, mean, logstd):
        sigma = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logstd)), max=10)
        noise = torch.randn(mean.size(0), self.opt["hidden_dim"], device=mean.device)
        if self.gc1.training:
            sampled_z = noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        user, user_kld = self._forward_user(ufea, UV_adj, VU_adj)
        item, item_kld = self._forward_item(vfea, UV_adj, VU_adj)
        self.kld_loss = user_kld + item_kld
        return user, item

    def _forward_user(self, ufea, UV_adj, VU_adj):
        h = self.gc1(ufea, VU_adj)
        mean = self.user_union_mean(torch.cat((self.gc3_mean(h, UV_adj), ufea), dim=1))
        logstd = self.user_union_logstd(torch.cat((self.gc3_logstd(h, UV_adj), ufea), dim=1))
        return self.reparameters(mean, logstd)

    def _forward_item(self, vfea, UV_adj, VU_adj):
        h = self.gc2(vfea, UV_adj)
        mean = self.item_union_mean(torch.cat((self.gc4_mean(h, VU_adj), vfea), dim=1))
        logstd = self.item_union_logstd(torch.cat((self.gc4_logstd(h, VU_adj), vfea), dim=1))
        return self.reparameters(mean, logstd)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        h = self.gc1(ufea, VU_adj)
        mean = self.user_union_mean(torch.cat((self.gc3_mean(h, UV_adj), ufea), dim=1))
        logstd = self.user_union_logstd(torch.cat((self.gc3_logstd(h, UV_adj), ufea), dim=1))
        return mean, logstd


class SingleVBGE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        layers = []
        for _ in range(opt["GNN"] - 1):
            layers.append(_SingleDGCNLayer(opt))
        layers.append(_SingleLastLayer(opt))
        self.encoder = nn.ModuleList(layers)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        u, v = ufea, vfea
        for layer in self.encoder:
            u = F.dropout(u, self.dropout, training=self.training)
            v = F.dropout(v, self.dropout, training=self.training)
            u, v = layer(u, v, UV_adj, VU_adj)
        return u, v

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        u = ufea
        for layer in self.encoder[:-1]:
            u = F.dropout(u, self.dropout, training=self.training)
            u = layer.forward_user_share(u, UV_adj, VU_adj)
        return self.encoder[-1].forward_user_share(u, UV_adj, VU_adj)


# ───────────────────── cross-domain VBGE ───────────────────────

class _CrossDGCNLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        fd, hd, dr, lk = opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"]
        self.gc1 = GCN(fd, hd, dr, lk)
        self.gc2 = GCN(fd, hd, dr, lk)
        self.gc3 = GCN(hd, fd, dr, lk)
        self.gc4 = GCN(hd, fd, dr, lk)
        self.source_user_union = nn.Linear(fd * 2, fd)
        self.target_user_union = nn.Linear(fd * 2, fd)
        self.register_buffer("source_rate", torch.tensor(opt["rate"]).view(-1))

    def forward(self, s_ufea, t_ufea, s_UV, s_VU, t_UV, t_VU):
        s_h = self.source_user_union(torch.cat((self.gc3(self.gc1(s_ufea, s_VU), s_UV), s_ufea), dim=1))
        t_h = self.target_user_union(torch.cat((self.gc4(self.gc2(t_ufea, t_VU), t_UV), t_ufea), dim=1))
        r = self.source_rate
        mixed = r * F.relu(s_h) + (1 - r) * F.relu(t_h)
        return mixed, mixed


class _CrossLastLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        fd, hd, dr, lk = opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"]
        self.gc1 = GCN(fd, hd, dr, lk)
        self.gc2 = GCN(fd, hd, dr, lk)
        self.gc3_mean = GCN(hd, fd, dr, lk)
        self.gc3_logstd = GCN(hd, fd, dr, lk)
        self.gc4_mean = GCN(hd, fd, dr, lk)
        self.gc4_logstd = GCN(hd, fd, dr, lk)
        self.source_user_union_mean = nn.Linear(fd * 2, fd)
        self.source_user_union_logstd = nn.Linear(fd * 2, fd)
        self.target_user_union_mean = nn.Linear(fd * 2, fd)
        self.target_user_union_logstd = nn.Linear(fd * 2, fd)
        self.register_buffer("source_rate", torch.tensor(opt["rate"]).view(-1))

    def forward(self, s_ufea, t_ufea, s_UV, s_VU, t_UV, t_VU):
        s_h = self.gc1(s_ufea, s_VU)
        t_h = self.gc2(t_ufea, t_VU)
        s_mean = self.source_user_union_mean(torch.cat((self.gc3_mean(s_h, s_UV), s_ufea), dim=1))
        s_logstd = self.source_user_union_logstd(torch.cat((self.gc3_logstd(s_h, s_UV), s_ufea), dim=1))
        t_mean = self.target_user_union_mean(torch.cat((self.gc4_mean(t_h, t_UV), t_ufea), dim=1))
        t_logstd = self.target_user_union_logstd(torch.cat((self.gc4_logstd(t_h, t_UV), t_ufea), dim=1))
        r = self.source_rate
        return r * s_mean + (1 - r) * t_mean, r * s_logstd + (1 - r) * t_logstd


class CrossVBGE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        layers = []
        for _ in range(opt["GNN"] - 1):
            layers.append(_CrossDGCNLayer(opt))
        layers.append(_CrossLastLayer(opt))
        self.encoder = nn.ModuleList(layers)
        self.dropout = opt["dropout"]

    def forward(self, s_ufea, t_ufea, s_UV, s_VU, t_UV, t_VU):
        s, t = s_ufea, t_ufea
        for layer in self.encoder[:-1]:
            s = F.dropout(s, self.dropout, training=self.training)
            t = F.dropout(t, self.dropout, training=self.training)
            s, t = layer(s, t, s_UV, s_VU, t_UV, t_VU)
        return self.encoder[-1](s, t, s_UV, s_VU, t_UV, t_VU)


# ───────────────── Adjacency matrix builder ─────────────────

def _row_normalize(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)


def _sp_to_torch(sp_mx, device):
    coo = sp_mx.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    val = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape)).coalesce().to(device)


def _build_uv_vu(train_inters, n_users, n_items, device):
    """Build row-normalized UV [n_users, n_items] and VU [n_items, n_users]."""
    rows, cols = [], []
    for inter in train_inters:
        rows.append(int(inter[0]))
        cols.append(int(inter[1]))
    rows, cols = np.array(rows), np.array(cols)
    UV = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(n_users, n_items), dtype=np.float32)
    VU = sp.coo_matrix((np.ones(len(rows)), (cols, rows)),
                        shape=(n_items, n_users), dtype=np.float32)
    UV = _row_normalize(UV)
    VU = _row_normalize(VU)
    return _sp_to_torch(UV, device), _sp_to_torch(VU, device)


# ═════════════════════════ DisenCDR Model ═════════════════════════

class DisenCDR(nn.Module):
    """DisenCDR adapted for the Copenhagen evaluation framework.

    - Uses unified user id space (n_users = d1_users + d2_users - shared).
    - Builds per-domain UV / VU bipartite adjacency matrices.
    - `calculate_loss` / `predict` follow Copenhagen conventions.
    - Warmup phase (first N epochs): only domain-specific GNNs, no cross-domain KLD.
    """

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device

        # ---- dataset dims (unified user space) ----
        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        # ---- hyper-params ----
        fd = params.get("disen_feature_dim", 128)
        hd = params.get("disen_hidden_dim", 128)
        n_gnn = params.get("disen_gnn_layers", 2)
        dropout = params.get("disen_dropout", 0.3)
        leakey = params.get("disen_leakey", 0.1)
        beta = params.get("disen_beta", 0.9)
        rate = params.get("disen_rate", 0.5)
        self.warmup_epochs = params.get("disen_warmup_epochs", 10)
        self.reg_weight = params.get("reg_weight", 0.0001)
        self.kld_weight = params.get("disen_kld_weight", 0.25)
        self.beta = beta

        opt = {
            "feature_dim": fd, "hidden_dim": hd, "GNN": n_gnn,
            "dropout": dropout, "leakey": leakey, "rate": rate,
        }

        # ---- sub-networks ----
        self.source_specific_GNN = SingleVBGE(opt)
        self.source_share_GNN = SingleVBGE(opt)
        self.target_specific_GNN = SingleVBGE(opt)
        self.target_share_GNN = SingleVBGE(opt)
        self.share_GNN = CrossVBGE(opt)

        # ---- embeddings (unified user space) ----
        self.source_user_emb = nn.Embedding(self.n_users, fd)
        self.target_user_emb = nn.Embedding(self.n_users, fd)
        self.source_item_emb = nn.Embedding(self.d1_n_items, fd)
        self.target_item_emb = nn.Embedding(self.d2_n_items, fd)
        self.source_user_share_emb = nn.Embedding(self.n_users, fd)
        self.target_user_share_emb = nn.Embedding(self.n_users, fd)

        self.share_mean = nn.Linear(fd * 2, fd)
        self.share_sigma = nn.Linear(fd * 2, fd)

        self._opt = opt
        self.kld_loss = 0
        self._epoch = [0]  # mutable container; updated externally via _epoch[0] = i

        # ---- build adjacency matrices ----
        d1_train_global = self._remap_d1(args.d1["train"])
        d2_train_global = self._remap_d2(args.d2["train"])
        self.d1_UV, self.d1_VU = _build_uv_vu(d1_train_global, self.n_users, self.d1_n_items, self.device)
        self.d2_UV, self.d2_VU = _build_uv_vu(d2_train_global, self.n_users, self.d2_n_items, self.device)

        self.apply(self._init_weights)

    # ── helpers ──

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _remap_d1(self, train):
        """d1 user ids are already 0..d1_n_users-1 in global space."""
        return train

    def _remap_d2(self, train):
        """Remap d2-local user ids to unified global space."""
        shift = self.d1_n_users - self.n_shared_users
        out = []
        for inter in train:
            u, i = int(inter[0]), int(inter[1])
            if u >= self.n_shared_users:
                u += shift
            out.append([u, i])
        return out

    def _d2_local_to_global(self, d2_users):
        """Map a tensor of d2-local user indices to global ids."""
        g = d2_users.clone()
        mask = d2_users >= self.n_shared_users
        g[mask] = d2_users[mask] + (self.d1_n_users - self.n_shared_users)
        return g

    def _global_to_d2_local(self, global_users):
        """Inverse of _d2_local_to_global: map global → d2-local."""
        out = global_users.clone()
        mask = global_users >= self.d1_n_users
        out[mask] = global_users[mask] - (self.d1_n_users - self.n_shared_users)
        return out

    # ── KLD utilities ──

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        sigma_1 = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logsigma_1)), max=10)
        sigma_2 = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logsigma_2)), max=10)
        return kl_divergence(Normal(mu_1, sigma_1), Normal(mu_2, sigma_2)).mean(dim=0).sum()

    def _reparameters(self, mean, logstd):
        sigma = torch.clamp(torch.exp(0.1 + 0.9 * F.softplus(logstd)), max=10)
        noise = torch.randn(mean.size(0), self._opt["hidden_dim"], device=mean.device)
        if self.training:
            sampled_z = noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.beta) * kld

    # ── forward (full / warmup) ──

    def _warmup_forward(self):
        s_user = self.source_user_emb.weight
        t_user = self.target_user_emb.weight
        s_item = self.source_item_emb.weight
        t_item = self.target_item_emb.weight

        s_user_out, s_item_out = self.source_specific_GNN(s_user, s_item, self.d1_UV, self.d1_VU)
        t_user_out, t_item_out = self.target_specific_GNN(t_user, t_item, self.d2_UV, self.d2_VU)
        self.kld_loss = 0
        return s_user_out, s_item_out, t_user_out, t_item_out

    def _full_forward(self):
        s_user = self.source_user_emb.weight
        t_user = self.target_user_emb.weight
        s_item = self.source_item_emb.weight
        t_item = self.target_item_emb.weight
        s_user_share = self.source_user_share_emb.weight
        t_user_share = self.target_user_share_emb.weight

        s_spec_user, s_spec_item = self.source_specific_GNN(s_user, s_item, self.d1_UV, self.d1_VU)
        t_spec_user, t_spec_item = self.target_specific_GNN(t_user, t_item, self.d2_UV, self.d2_VU)

        s_share_mean, s_share_sigma = self.source_share_GNN.forward_user_share(s_user, self.d1_UV, self.d1_VU)
        t_share_mean, t_share_sigma = self.target_share_GNN.forward_user_share(t_user, self.d2_UV, self.d2_VU)

        cross_mean, cross_sigma = self.share_GNN(
            s_user_share, t_user_share, self.d1_UV, self.d1_VU, self.d2_UV, self.d2_VU)

        user_share, share_kld = self._reparameters(cross_mean, cross_sigma)

        s_share_kld = self._kld_gauss(cross_mean, cross_sigma, s_share_mean, s_share_sigma)
        t_share_kld = self._kld_gauss(cross_mean, cross_sigma, t_share_mean, t_share_sigma)

        self.kld_loss = (
            share_kld
            + self.beta * s_share_kld
            + self.beta * t_share_kld
        )

        s_user_out = user_share + s_spec_user
        t_user_out = user_share + t_spec_user
        return s_user_out, s_spec_item, t_user_out, t_spec_item

    def forward(self):
        if self._epoch[0] < self.warmup_epochs:
            return self._warmup_forward()
        return self._full_forward()

    # ── scoring ──

    def get_score(self, user_emb, item_emb):
        return (user_emb * item_emb).sum(dim=-1)

    def get_bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos = self.get_score(user_emb, pos_item_emb)
        neg = self.get_score(user_emb, neg_item_emb)
        return -F.logsigmoid(pos - neg).mean()

    def get_reg_loss(self, *embs):
        loss = sum(e.norm(2).pow(2) for e in embs)
        return loss / (2 * embs[0].shape[0])

    # ── training loss (Copenhagen interface) ──

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        s_user, s_item, t_user, t_item = self.forward()

        # d1: user ids already global (0..d1_n_users-1)
        d1_u = s_user[inter_d1[:, 0]]
        d1_pos = s_item[inter_d1[:, 1]]
        d1_neg = s_item[neg_d1_item]

        # d2: remap local user ids → global
        d2_global_users = self._d2_local_to_global(inter_d2[:, 0])
        d2_u = t_user[d2_global_users]
        d2_pos = t_item[inter_d2[:, 1]]
        d2_neg = t_item[neg_d2_item]

        d1_bpr = self.get_bpr_loss(d1_u, d1_pos, d1_neg)
        d2_bpr = self.get_bpr_loss(d2_u, d2_pos, d2_neg)

        d1_reg = self.reg_weight * self.get_reg_loss(
            self.source_user_emb.weight[inter_d1[:, 0]],
            self.source_item_emb.weight[inter_d1[:, 1]],
            self.source_item_emb.weight[neg_d1_item],
        )
        d2_reg = self.reg_weight * self.get_reg_loss(
            self.target_user_emb.weight[d2_global_users],
            self.target_item_emb.weight[inter_d2[:, 1]],
            self.target_item_emb.weight[neg_d2_item],
        )

        spec_kld = (
            self.source_specific_GNN.encoder[-1].kld_loss
            + self.target_specific_GNN.encoder[-1].kld_loss
        )

        loss = (d1_bpr + d2_bpr) / 2 + (d1_reg + d2_reg) / 2 + self.kld_weight * (spec_kld + self.kld_loss)
        return loss

    # ── evaluation (Copenhagen interface) ──

    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history, is_d2=False):
        """Full in-domain ranking with user-history masking.

        ``user_emb`` is in the unified user space; for domain-2 the local user
        ids must be remapped before indexing.
        """
        def score_fn(users):
            if is_d2:
                users = self._d2_local_to_global(users)
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        s_user, s_item, t_user, t_item = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, s_user, s_item, self.d1_domain_name, mode, mask_d1, is_d2=False)
        if domain == 2:
            return self._eval_domain(eval_set_2, t_user, t_item, self.d2_domain_name, mode, mask_d2, is_d2=True)
        d1 = self._eval_domain(eval_set_1, s_user, s_item, self.d1_domain_name, mode, mask_d1, is_d2=False)
        d2 = self._eval_domain(eval_set_2, t_user, t_item, self.d2_domain_name, mode, mask_d2, is_d2=True)
        return [d1, d2]
