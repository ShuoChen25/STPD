import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import full_rank_eval


class _CrossDomainNN(nn.Module):

    def __init__(self, emb_dim, hidden_dim, dropout, n_layers):
        super().__init__()
        layers = [nn.Linear(2 * emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u_t, u_s):
        return self.net(torch.cat([u_t, u_s], dim=1))


class CDFA(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = int(params.get("embedding_size", 64))
        self.alpha = float(params.get("cdfa_alpha", 0.1))
        self.lamda = float(params.get("cdfa_lambda", 0.1))
        self.gamma = float(params.get("cdfa_gamma", 0.1))
        self.nn_hidden = int(params.get("cdfa_nn_hidden", self.emb_dim))
        self.nn_dropout = float(params.get("cdfa_nn_dropout", 0.2))
        self.nn_layers = int(params.get("cdfa_nn_layers", 3))
        self.nn_weight = float(params.get("cdfa_nn_weight", 1e-5))
        self.fairness_method = str(params.get("cdfa_fairness_method", "wasserstein"))
        self.fairness_weight = float(params.get("cdfa_fairness_weight", 0.05))

        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.d1_name = args.d1.get("name", "domain1").lower()
        self.d2_name = args.d2.get("name", "domain2").lower()

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        self.cross_domain_nn = _CrossDomainNN(
            self.emb_dim, self.nn_hidden, self.nn_dropout, self.nn_layers
        )

        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        groups = self._build_user_groups(args.d2["train"])
        self.register_buffer("user_groups", torch.tensor(groups, dtype=torch.long))

        self.register_buffer(
            "prev_overlap_user_emb",
            torch.zeros(self.n_shared_users, self.emb_dim, dtype=self.dtype),
        )
        self.has_prev_snapshot = False
        self._epoch = [0]
        self._last_snapshot_epoch = -1

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_user_groups(self, d2_train):
        counts = np.zeros(self.n_users, dtype=np.int64)
        offset = self.d1_n_users - self.n_shared_users
        for row in d2_train:
            u = row[0]
            uu = u if u < self.n_shared_users else u + offset
            counts[uu] += 1
        active = counts > 0
        groups = np.zeros(self.n_users, dtype=np.int64)
        if active.any():
            threshold = float(np.median(counts[active]))
            groups[active & (counts >= threshold)] = 1
            groups[active & (counts < threshold)] = 2
        return groups


    def _d2_to_union(self, d2_user_ids):
        return torch.where(
            d2_user_ids < self.n_shared_users,
            d2_user_ids,
            d2_user_ids + (self.d1_n_users - self.n_shared_users),
        )

    def _maybe_snapshot_prev_emb(self):
        cur_epoch = int(self._epoch[0])
        if cur_epoch != self._last_snapshot_epoch:
            with torch.no_grad():
                ids = torch.arange(self.n_shared_users, device=self.user_embedding.weight.device)
                self.prev_overlap_user_emb.copy_(self.user_embedding(ids).detach())
            if self._last_snapshot_epoch >= 0:
                self.has_prev_snapshot = True
            self._last_snapshot_epoch = cur_epoch

    def _bce_pair(self, u_emb, pos_i_emb, neg_i_emb):
        if u_emb.shape[0] == 0:
            return u_emb.new_zeros(())
        pos = self.sigmoid((u_emb * pos_i_emb).sum(-1))
        neg = self.sigmoid((u_emb * neg_i_emb).sum(-1))
        return self.bce(pos, torch.ones_like(pos)) + self.bce(neg, torch.zeros_like(neg))

    @staticmethod
    def _emb_reg(*embs):
        if embs[0].shape[0] == 0:
            return embs[0].new_zeros(())
        loss = sum((e ** 2).sum() for e in embs) / 2
        return loss / embs[0].shape[0]


    def _fairness_loss(self, user_union_ids, u_emb, pos_i_emb):
        p_xs_xt = self.sigmoid((self.cross_domain_nn(u_emb, u_emb) * pos_i_emb).sum(-1))
        p_xs = self.sigmoid((u_emb * pos_i_emb).sum(-1))
        p_xt = p_xs

        groups = self.user_groups[user_union_ids]
        m1 = groups == 1
        m2 = groups == 2
        if m1.sum() == 0 or m2.sum() == 0:
            return u_emb.new_zeros(())

        d1 = p_xs_xt[m1] / (p_xs[m1] * p_xt[m1] + 1e-8)
        d2 = p_xs_xt[m2] / (p_xs[m2] * p_xt[m2] + 1e-8)

        method = self.fairness_method
        if method == "kl_div":
            h1 = torch.histc(d1, bins=50, min=0, max=2)
            h2 = torch.histc(d2, bins=50, min=0, max=2)
            h1 = h1 / (h1.sum() + 1e-8)
            h2 = h2 / (h2.sum() + 1e-8)
            return (h1 * torch.log((h1 + 1e-8) / (h2 + 1e-8))).sum()
        if method == "wasserstein":
            s1, _ = torch.sort(d1)
            s2, _ = torch.sort(d2)
            n = max(s1.numel(), s2.numel())
            if s1.numel() < n:
                s1 = torch.cat([s1, s1[-1].expand(n - s1.numel())])
            if s2.numel() < n:
                s2 = torch.cat([s2, s2[-1].expand(n - s2.numel())])
            return (s1 - s2).abs().mean()
        if method == "moment_matching":
            return (d1.mean() - d2.mean()).pow(2) + 0.1 * (
                d1.var(unbiased=False) - d2.var(unbiased=False)
            ).pow(2)
        return (d1.mean() - d2.mean()).pow(2)


    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        self._maybe_snapshot_prev_emb()

        d1_u_idx = inter_d1[:, 0]
        d1_pos_idx = inter_d1[:, 1]
        d1_u = self.user_embedding(d1_u_idx)
        d1_pos = self.d1_item(d1_pos_idx)
        d1_neg = self.d1_item(neg_d1_item)
        loss_s = self._bce_pair(d1_u, d1_pos, d1_neg) + self.lamda * self._emb_reg(d1_u, d1_pos, d1_neg)

        d2_u_local = inter_d2[:, 0]
        d2_pos_idx = inter_d2[:, 1]
        d2_u_union = self._d2_to_union(d2_u_local)
        d2_u = self.user_embedding(d2_u_union)
        d2_pos = self.d2_item(d2_pos_idx)
        d2_neg = self.d2_item(neg_d2_item)
        loss_t = self._bce_pair(d2_u, d2_pos, d2_neg) + self.gamma * self._emb_reg(d2_u, d2_pos, d2_neg)

        overlap_mask = d2_u_union < self.n_shared_users
        fairness_loss = d2_u.new_zeros(())
        nn_loss = d2_u.new_zeros(())
        if overlap_mask.any():
            ov_user = d2_u_union[overlap_mask]
            ov_u = d2_u[overlap_mask]
            ov_i = d2_pos[overlap_mask]
            fairness_loss = self._fairness_loss(ov_user, ov_u, ov_i)
            if self.has_prev_snapshot:
                prev = self.prev_overlap_user_emb[ov_user]
                enhanced = self.cross_domain_nn(prev, prev)
                nn_loss = self.nn_weight * F.mse_loss(enhanced, ov_u.detach())

        return self.alpha * loss_s + (1.0 - self.alpha) * loss_t \
            + self.fairness_weight * fairness_loss + nn_loss


    def _d2_user_view(self):
        emb = self.user_embedding.weight
        return torch.cat([emb[: self.n_shared_users], emb[self.d1_n_users: self.n_users]], dim=0)

    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user = self.user_embedding.weight[: self.d1_n_users]
        d1_item = self.d1_item.weight
        d2_user = self._d2_user_view()
        d2_item = self.d2_item.weight
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_name, mode, mask_d2)
        d1 = self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_name, mode, mask_d1)
        d2 = self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_name, mode, mask_d2)
        return [d1, d2]
