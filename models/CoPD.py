import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval, symmetric_normalize


class CoPD(nn.Module):
    """CoPD (Consistency-guided Popularity Disentanglement) adapted for STPD framework.

    Three GCN branches:
      - d1_graph: domain-1 specific
      - d2_graph: domain-2 specific
      - cross_graph: shared-user ↔ all-items (both domains)

    Losses:
      1. BPR recommendation loss (both domains)
      2. Consistency constraint (cc) — aligns item spe/share embeddings
      3. Domain classifier (dom) — disentangles shared vs specific user embs
      4. Preference disentanglement (pd) — separates popularity from interest
    """

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params["embedding_size"]
        self.intra_domain_layers = params.get("intra_domain_layers", 3)
        self.drop_rate = params.get("drop_rate", 0.3)
        self.reg_weight = params.get("reg_weight", 0.0001)
        self.temp = params.get("temp", 0.1)
        self.lambda1 = params.get("lambda1", 0.01)
        self.lambda2 = params.get("lambda2", 0.5)

        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.neg_ratio = params.get("neg_ratio", 1)

        self.dropout = nn.Dropout(self.drop_rate)

        # Domain-specific user/item embeddings (global user space)
        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)
        # Shared user embedding (only shared users)
        self.share_user = nn.Embedding(self.n_shared_users, self.emb_dim, dtype=self.dtype)

        # Domain-specific graphs (already built by load_data → get_inter_mat)
        self.d1_graph = self._build_norm_graph(args.d1["inter_mat"])
        self.d2_graph = self._build_norm_graph(args.d2["inter_mat"])
        # Cross-domain graph: shared users ↔ items from both domains
        self.cross_graph = self._build_cross_graph(args)

        # Popularity / Interest encoders
        self.pop_encoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(self.drop_rate),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.int_encoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(self.drop_rate),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        self.domain_cls = nn.Linear(self.emb_dim, 2)

        # Attention-based user embedding fusion  (spe, int, pop) → fused
        self.att1_d1 = nn.Linear(self.emb_dim * 3, self.emb_dim)
        self.att2_d1 = nn.Linear(self.emb_dim, 3)
        self.att1_d2 = nn.Linear(self.emb_dim * 3, self.emb_dim)
        self.att2_d2 = nn.Linear(self.emb_dim, 3)

        # Item aggregation  (spe ‖ share) → fused
        self.agg_d1 = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.agg_d2 = nn.Linear(self.emb_dim * 2, self.emb_dim)

        # Loss functions
        self.loss_cos = nn.CosineEmbeddingLoss()
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_KLD = nn.KLDivLoss(reduction="batchmean")

        # Precompute item popularity (normalised to [0, 1])
        self._precompute_popularity(args)

        # Xavier init (embeddings only, matching original CoPD)
        self._init_weights()

    # ------------------------------------------------------------------ init helpers

    def _init_weights(self):
        nn.init.xavier_uniform_(self.d1_user.weight)
        nn.init.xavier_uniform_(self.d1_item.weight)
        nn.init.xavier_uniform_(self.d2_user.weight)
        nn.init.xavier_uniform_(self.d2_item.weight)
        nn.init.xavier_uniform_(self.share_user.weight)

    def _build_norm_graph(self, sp_mat):
        norm = symmetric_normalize(sp_mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _build_cross_graph(self, args):
        """Shared-user ↔ all-item bipartite graph.

        Node layout: [shared_users | d1_items | d2_items]
        Size: (n_shared + d1_items + d2_items) ^ 2
        """
        n_sh = self.n_shared_users
        total = n_sh + self.d1_n_items + self.d2_n_items

        edges = []
        for user, item, *_ in args.d1["train"]:
            if user < n_sh:
                j = n_sh + item
                edges.append((user, j))
                edges.append((j, user))
        for user, item, *_ in args.d2["train"]:
            if user < n_sh:
                j = n_sh + self.d1_n_items + item
                edges.append((user, j))
                edges.append((j, user))

        if not edges:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=self.dtype),
                (total, total),
            ).to(self.device)

        edges = np.array(edges)
        data = np.ones(len(edges))
        sp_mat = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(total, total))
        return self._build_norm_graph(sp_mat)

    def _precompute_popularity(self, args):
        d1_pop = torch.zeros(self.d1_n_items, dtype=self.dtype)
        for _, item, *_ in args.d1["train"]:
            d1_pop[item] += 1
        d1_pop = d1_pop / d1_pop.max().clamp(min=1)

        d2_pop = torch.zeros(self.d2_n_items, dtype=self.dtype)
        for _, item, *_ in args.d2["train"]:
            d2_pop[item] += 1
        d2_pop = d2_pop / d2_pop.max().clamp(min=1)

        self.register_buffer("d1_item_pop", d1_pop)
        self.register_buffer("d2_item_pop", d2_pop)

    # ------------------------------------------------------------------ GCN

    def aggregate(self, graph, all_emb, n_layers):
        embeddings = [all_emb]
        for _ in range(n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            all_emb = self.dropout(all_emb)
            embeddings.append(all_emb)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    # ------------------------------------------------------------------ split

    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[: self.d1_n_users], emb[self.n_users :]
        user = torch.cat(
            [emb[: self.n_shared_users], emb[self.d1_n_users : self.n_users]], dim=0
        )
        return user, emb[self.n_users :]

    # ------------------------------------------------------------------ fusion

    def _emb_fusion(self, spe, int_emb, pop_emb, domain):
        att1 = self.att1_d1 if domain == 1 else self.att1_d2
        att2 = self.att2_d1 if domain == 1 else self.att2_d2
        x = torch.cat([spe, int_emb, pop_emb], dim=1)
        x = F.relu(att1(x))
        x = F.dropout(x, training=self.training, p=self.drop_rate)
        w = F.softmax(att2(x), dim=1)
        w1, w2, w3 = w.chunk(3, dim=1)
        return spe * w1 + int_emb * w2 + pop_emb * w3

    # ------------------------------------------------------------------ forward

    def forward(self):
        # --- domain-specific GCN ---
        d1_emb = self.aggregate(
            self.d1_graph, torch.cat([self.d1_user.weight, self.d1_item.weight]), self.intra_domain_layers
        )
        d2_emb = self.aggregate(
            self.d2_graph, torch.cat([self.d2_user.weight, self.d2_item.weight]), self.intra_domain_layers
        )
        d1_user_spe, d1_item_spe = self._split_domain_ui(d1_emb, 1)
        d2_user_spe, d2_item_spe = self._split_domain_ui(d2_emb, 2)

        # --- cross-domain GCN ---
        cross_emb = self.aggregate(
            self.cross_graph,
            torch.cat([self.share_user.weight, self.d1_item.weight, self.d2_item.weight]),
            self.intra_domain_layers,
        )
        user_sha = cross_emb[: self.n_shared_users]
        item_sha_d1 = cross_emb[self.n_shared_users : self.n_shared_users + self.d1_n_items]
        item_sha_d2 = cross_emb[self.n_shared_users + self.d1_n_items :]

        # --- disentangle shared user preference ---
        pop_user = self.pop_encoder(user_sha)
        int_user = self.int_encoder(user_sha)

        # --- fuse user embeddings (shared users only; others keep spe) ---
        d1_user_fused = d1_user_spe.clone()
        d1_user_fused[: self.n_shared_users] = self._emb_fusion(
            d1_user_spe[: self.n_shared_users], int_user, pop_user, domain=1
        )
        d2_user_fused = d2_user_spe.clone()
        d2_user_fused[: self.n_shared_users] = self._emb_fusion(
            d2_user_spe[: self.n_shared_users], int_user, pop_user, domain=2
        )

        # --- fuse item embeddings (spe + share) ---
        d1_item_fused = self.agg_d1(torch.cat([d1_item_spe, item_sha_d1], dim=1))
        d2_item_fused = self.agg_d2(torch.cat([d2_item_spe, item_sha_d2], dim=1))

        return (
            d1_user_fused, d1_item_fused,
            d2_user_fused, d2_item_fused,
            # auxiliaries for CoPD losses (ignored by predict)
            d1_user_spe, d2_user_spe,
            user_sha, pop_user, int_user,
            d1_item_spe, d2_item_spe,
            item_sha_d1, item_sha_d2,
        )

    # ------------------------------------------------------------------ scoring

    def get_score(self, user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)

    # ------------------------------------------------------------------ loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        (
            d1_user, d1_item,
            d2_user, d2_item,
            d1_user_spe, d2_user_spe,
            user_sha, pop_user, int_user,
            d1_item_spe, d2_item_spe,
            item_sha_d1, item_sha_d2,
        ) = self.forward()

        # ---- 1. BPR recommendation loss ----
        d1_u = d1_user[inter_d1[:, 0]]
        d1_pos = d1_item[inter_d1[:, 1]]
        d2_u = d2_user[inter_d2[:, 0]]
        d2_pos = d2_item[inter_d2[:, 1]]

        d1_loss, d2_loss = 0.0, 0.0
        for i in range(self.neg_ratio):
            d1_neg = d1_item[neg_d1_item[:, i]]
            d1_loss += -F.logsigmoid(self.get_score(d1_u, d1_pos) - self.get_score(d1_u, d1_neg)).mean()
            d2_neg = d2_item[neg_d2_item[:, i]]
            d2_loss += -F.logsigmoid(self.get_score(d2_u, d2_pos) - self.get_score(d2_u, d2_neg)).mean()
        d1_loss /= self.neg_ratio
        d2_loss /= self.neg_ratio

        # ego embedding regularisation
        reg = (
            self.d1_user.weight[inter_d1[:, 0]].norm(2).pow(2)
            + self.d1_item.weight[inter_d1[:, 1]].norm(2).pow(2)
            + self.d1_item.weight[neg_d1_item[:, 0]].norm(2).pow(2)
            + self.d2_user.weight[inter_d2[:, 0]].norm(2).pow(2)
            + self.d2_item.weight[inter_d2[:, 1]].norm(2).pow(2)
            + self.d2_item.weight[neg_d2_item[:, 0]].norm(2).pow(2)
        )
        sh_users = torch.unique(torch.cat([
            inter_d1[:, 0][inter_d1[:, 0] < self.n_shared_users],
            inter_d2[:, 0][inter_d2[:, 0] < self.n_shared_users],
        ]))
        if sh_users.numel():
            reg = reg + self.share_user.weight[sh_users].norm(2).pow(2)
        reg = reg / (2 * (len(inter_d1) + len(inter_d2)))

        loss_rec = d1_loss + d2_loss + self.reg_weight * reg

        # ---- CoPD auxiliary losses (shared users only) ----
        d1_sh = inter_d1[:, 0] < self.n_shared_users
        d2_sh = inter_d2[:, 0] < self.n_shared_users

        if not (d1_sh.any() and d2_sh.any()):
            return loss_rec

        d1_sh_u = inter_d1[d1_sh, 0]
        d1_sh_pos = inter_d1[d1_sh, 1]
        d1_sh_neg = neg_d1_item[d1_sh, 0]
        d2_sh_u = inter_d2[d2_sh, 0]
        d2_sh_pos = inter_d2[d2_sh, 1]
        d2_sh_neg = neg_d2_item[d2_sh, 0]
        n1, n2 = d1_sh_u.shape[0], d2_sh_u.shape[0]

        # ---- 2. Consistency constraint loss ----
        pos_lbl1 = torch.ones(n1, device=self.device, dtype=torch.long)
        neg_lbl1 = -pos_lbl1
        loss_cc = (
            self.loss_cos(d1_item_spe[d1_sh_pos].detach(), item_sha_d1[d1_sh_pos], pos_lbl1)
            + self.loss_cos(d1_item_spe[d1_sh_neg].detach(), item_sha_d1[d1_sh_pos], neg_lbl1)
        )
        pos_lbl2 = torch.ones(n2, device=self.device, dtype=torch.long)
        neg_lbl2 = -pos_lbl2
        loss_cc = loss_cc + (
            self.loss_cos(d2_item_spe[d2_sh_pos].detach(), item_sha_d2[d2_sh_pos], pos_lbl2)
            + self.loss_cos(d2_item_spe[d2_sh_neg].detach(), item_sha_d2[d2_sh_pos], neg_lbl2)
        )

        # ---- 3. Domain classifier loss ----
        if n1 == n2 and torch.equal(d1_sh_u, d2_sh_u):
            sh_users = d1_sh_u
        else:
            sh_users = torch.cat([d1_sh_u, d2_sh_u], dim=0)
        share_score = self.domain_cls(user_sha[sh_users])
        y_uniform = torch.ones(sh_users.shape[0], 2, device=self.device) / 2.0
        loss_dom = self.loss_KLD(F.log_softmax(share_score, dim=1), y_uniform)
        loss_dom = loss_dom + self.loss_CE(
            self.domain_cls(d1_user_spe[d1_sh_u]),
            torch.ones(n1, device=self.device, dtype=torch.long),
        )
        loss_dom = loss_dom + self.loss_CE(
            self.domain_cls(d2_user_spe[d2_sh_u]),
            torch.zeros(n2, device=self.device, dtype=torch.long),
        )

        # ---- 4. Preference disentanglement loss ----
        pop_u1, int_u1 = pop_user[d1_sh_u], int_user[d1_sh_u]
        pop_u2, int_u2 = pop_user[d2_sh_u], int_user[d2_sh_u]
        sha_pos_d1 = item_sha_d1[d1_sh_pos]
        sha_pos_d2 = item_sha_d2[d2_sh_pos]

        pop_d1 = self.d1_item_pop[d1_sh_pos].unsqueeze(1)
        pop_d2 = self.d2_item_pop[d2_sh_pos].unsqueeze(1)

        def _pd_loss(pop_u, int_u, sha_pos, pop_w, item_sha_all):
            cw = torch.exp(pop_w)
            iw = torch.exp(1.0 - pop_w)
            pos_pop = self.get_score(pop_u, sha_pos)
            pos_int = self.get_score(int_u, sha_pos)
            l_conf = (
                -torch.log(cw * torch.exp(pos_pop / self.temp)).mean()
                + torch.log(torch.exp(pop_u @ item_sha_all.T / self.temp).sum(1) + 1e-8).mean()
            )
            l_int = (
                -torch.log(iw * torch.exp(pos_int / self.temp)).mean()
                + torch.log(torch.exp(int_u @ item_sha_all.T / self.temp).sum(1) + 1e-8).mean()
            )
            return l_conf + l_int

        loss_pd = _pd_loss(pop_u1, int_u1, sha_pos_d1, pop_d1, item_sha_d1)
        loss_pd = loss_pd + _pd_loss(pop_u2, int_u2, sha_pos_d2, pop_d2, item_sha_d2)

        return loss_rec + loss_dom + self.lambda1 * loss_cc + self.lambda2 * loss_pd

    # ------------------------------------------------------------------ evaluation

    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        """Full in-domain ranking with user-history masking."""
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user, d1_item, d2_user, d2_item, *_ = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        d1_m = self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        d2_m = self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        return [d1_m, d2_m]
