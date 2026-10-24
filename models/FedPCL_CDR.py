
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval, symmetric_normalize


def _kmeans(x, k, n_iter=20, seed=42):
    n = x.size(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(n_iter):
        d = torch.cdist(x, centers)
        new_labels = d.argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = x[mask].mean(dim=0)
            else:
                far_idx = d.gather(1, labels.unsqueeze(1)).squeeze(1).argmax()
                centers[c] = x[far_idx]
                labels[far_idx] = c
    return labels, centers


class FedPCL_CDR(nn.Module):
    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params.get("embedding_size", 64)
        self.n_layers = params.get("fedpcl_n_layers", 3)
        self.num_clusters = params.get("fedpcl_num_clusters", 10)
        self.tau = params.get("fedpcl_tau", 0.2)
        self.alpha = params.get("fedpcl_alpha", 0.01)
        self.lap_noise = params.get("fedpcl_lap_noise", 0.1)
        self.clip_C = params.get("fedpcl_clip_C", 0.2)
        self.reg_weight = params.get("fedpcl_reg_weight", params.get("reg_weight", 1e-4))

        self.n_shared = args.data["n_shared_users"]
        self.client_n_users = [args.d1["n_users"], args.d2["n_users"]]
        self.client_n_items = [args.d1["n_items"], args.d2["n_items"]]
        self.client_names = [
            args.d1.get("name", "domain1").lower(),
            args.d2.get("name", "domain2").lower(),
        ]

        self.user_embs = nn.ModuleList([
            nn.Embedding(self.client_n_users[c], self.emb_dim, dtype=self.dtype)
            for c in range(2)
        ])
        self.item_embs = nn.ModuleList([
            nn.Embedding(self.client_n_items[c], self.emb_dim, dtype=self.dtype)
            for c in range(2)
        ])

        self.graphs = [
            self._build_local_graph(args.d1["train"], self.client_n_users[0], self.client_n_items[0]),
            self._build_local_graph(args.d2["train"], self.client_n_users[1], self.client_n_items[1]),
        ]

        def _make_classifier():
            d = self.emb_dim
            return nn.Sequential(
                nn.Linear(d * 2, d), nn.BatchNorm1d(d), nn.ReLU(),
                nn.Linear(d, d // 2), nn.BatchNorm1d(d // 2), nn.ReLU(),
                nn.Linear(d // 2, max(d // 4, 1)), nn.BatchNorm1d(max(d // 4, 1)), nn.ReLU(),
                nn.Linear(max(d // 4, 1), 1),
            )
        self.classifiers = nn.ModuleList([_make_classifier() for _ in range(2)])

        self.bce = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)

        self.user2cat = [None, None]
        self.proto_C = [None, None]
        self.proto_P0 = [None, None]
        self.proto_P1 = [None, None]

        self._epoch = [0]
        self._last_proto_epoch = [-1]


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def _build_local_graph(self, train_pairs, n_users, n_items):
        rows, cols = [], []
        for inter in train_pairs:
            u, i = inter[0], inter[1] + n_users
            rows.append(u); cols.append(i)
            rows.append(i); cols.append(u)
        n = n_users + n_items
        data = np.ones(len(rows), dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        norm = symmetric_normalize(mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)


    def _client_aggregate(self, c):
        u_w = self.user_embs[c].weight
        i_w = self.item_embs[c].weight
        all_emb = torch.cat([u_w, i_w], dim=0)
        layers = [all_emb]
        graph = self.graphs[c]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            layers.append(all_emb)
        agg = torch.stack(layers, dim=1).sum(dim=1)
        nu = self.client_n_users[c]
        return agg[:nu], agg[nu:]


    @torch.no_grad()
    def _federated_round(self):
        K = self.num_clusters
        D = self.emb_dim
        local = [None, None]

        for c in range(2):
            user_emb, _ = self._client_aggregate(c)
            labels_t, centers_t = _kmeans(user_emb.detach(), K, n_iter=20, seed=42)
            labels_np = labels_t.cpu().numpy().astype(np.int64)
            self.user2cat[c] = labels_np

            cluster = {}
            for cat in range(K):
                centroid = centers_t[cat]
                l2 = float(centroid.norm().item())
                scale = max(1.0, l2 / self.clip_C)
                clipped = centroid / scale
                noise = torch.from_numpy(
                    np.random.laplace(0.0, self.lap_noise, clipped.shape).astype(np.float32)
                ).to(self.device)
                anchor = clipped + noise
                indices = np.where(labels_np[: self.n_shared] == cat)[0].tolist()
                cluster[cat] = {"anchor": anchor, "indices": indices}
            local[c] = cluster

        for j in range(2):
            P0 = torch.zeros(K, D, device=self.device)
            P1 = torch.zeros(K, D, device=self.device)
            Cg = torch.zeros(K, D, device=self.device)
            for cat in range(K):
                anchor = local[j][cat]["anchor"]
                indices = local[j][cat]["indices"]
                if not indices:
                    P0[cat] = anchor; P1[cat] = anchor; Cg[cat] = anchor
                    continue
                best = [None, None]
                pool = []
                for k in range(2):
                    overlap_cats = {int(self.user2cat[k][u]) for u in indices}
                    best_sim, best_anchor = -float("inf"), anchor
                    for ck in overlap_cats:
                        ak = local[k][ck]["anchor"]
                        pool.append(ak)
                        s = F.cosine_similarity(anchor.unsqueeze(0), ak.unsqueeze(0)).item()
                        if s > best_sim:
                            best_sim, best_anchor = s, ak
                    best[k] = best_anchor
                P0[cat] = best[0]
                P1[cat] = best[1]
                Cg[cat] = torch.stack(pool, dim=0).mean(dim=0) if pool else anchor
            self.proto_C[j] = Cg
            self.proto_P0[j] = P0
            self.proto_P1[j] = P1


    def _proto_loss(self, c, u_feats, cats):
        device = self.device
        K = self.num_clusters
        u_n = F.normalize(u_feats, dim=1)
        P0_n = F.normalize(self.proto_P0[c], dim=1)
        P1_n = F.normalize(self.proto_P1[c], dim=1)
        C_n = F.normalize(self.proto_C[c], dim=1)

        sim_p0 = (u_n @ P0_n.t()) / self.tau
        sim_p1 = (u_n @ P1_n.t()) / self.tau
        sim_c = (u_n @ C_n.t()) / self.tau

        B = u_feats.size(0)
        mask = torch.ones(B, 2 * K, device=device, dtype=torch.bool)
        rows = torch.arange(B, device=device)
        mask[rows, cats] = False
        mask[rows, cats + K] = False
        all_logits = torch.cat([sim_p0, sim_p1], dim=1)
        neg_logits = all_logits[mask].view(B, 2 * K - 2)

        pos_p0 = sim_p0.gather(1, cats.unsqueeze(1)).squeeze(1)
        pos_p1 = sim_p1.gather(1, cats.unsqueeze(1)).squeeze(1)

        denom_p0 = torch.logsumexp(torch.cat([pos_p0.unsqueeze(1), neg_logits], dim=1), dim=1)
        denom_p1 = torch.logsumexp(torch.cat([pos_p1.unsqueeze(1), neg_logits], dim=1), dim=1)
        log_prob_1 = pos_p0 - denom_p0
        log_prob_2 = pos_p1 - denom_p1
        L_P = -torch.logaddexp(log_prob_1, log_prob_2).mean()

        L_C = F.cross_entropy(sim_c, cats)
        return L_C, L_P


    def _maybe_run_federated(self):
        cur = int(self._epoch[0])
        if cur != self._last_proto_epoch[0]:
            if self._last_proto_epoch[0] >= 0:
                self._federated_round()
            self._last_proto_epoch[0] = cur

    def _client_step_loss(self, c, inter, neg_item):
        users = inter[:, 0]
        pos_items = inter[:, 1]

        all_users = torch.cat([users, users])
        all_items = torch.cat([pos_items, neg_item])
        labels = torch.cat([
            torch.ones(users.size(0), dtype=torch.float32, device=self.device),
            torch.zeros(users.size(0), dtype=torch.float32, device=self.device),
        ])

        u_emb_all, i_emb_all = self._client_aggregate(c)
        u_f = u_emb_all[all_users]
        v_f = i_emb_all[all_items]
        logits = self.classifiers[c](torch.cat([u_f, v_f], dim=1)).squeeze(-1)
        L = self.bce(logits, labels)

        if self.proto_C[c] is not None and self.alpha > 0:
            cat_arr = self.user2cat[c][users.detach().cpu().numpy()]
            cats = torch.from_numpy(cat_arr).long().to(self.device)
            L_C, L_P = self._proto_loss(c, u_emb_all[users], cats)
            L = L + self.alpha * (L_C + L_P)

        if self.reg_weight > 0:
            reg = (u_f.pow(2).sum() + v_f.pow(2).sum()) / (2 * u_f.size(0))
            L = L + self.reg_weight * reg
        return L

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        self._maybe_run_federated()
        L1 = self._client_step_loss(0, inter_d1, neg_d1_item)
        L2 = self._client_step_loss(1, inter_d2, neg_d2_item)
        return (L1 + L2) / 2


    def _eval_domain(self, eval_set, c, name, mode, history):
        u_emb_all, i_emb_all = self._client_aggregate(c)
        n_items = self.client_n_items[c]
        item_emb = i_emb_all[:n_items]
        device = item_emb.device
        item_chunk = 512

        def score_fn(users):
            users = users.long()
            u_f = u_emb_all[users]
            B = u_f.size(0)
            scores = torch.empty(B, n_items, device=device)
            for s in range(0, n_items, item_chunk):
                e = min(n_items, s + item_chunk)
                C = e - s
                v_chunk = item_emb[s:e]
                uu = u_f.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
                vv = v_chunk.unsqueeze(0).expand(B, C, -1).reshape(B * C, -1)
                logits = self.classifiers[c](torch.cat([uu, vv], dim=1)).squeeze(-1)
                scores[:, s:e] = logits.view(B, C)
            return scores

        return full_rank_eval(eval_set, score_fn, history, name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 0, self.client_names[0], mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, 1, self.client_names[1], mode, mask_d2)
        d1 = self._eval_domain(eval_set_1, 0, self.client_names[0], mode, mask_d1)
        d2 = self._eval_domain(eval_set_2, 1, self.client_names[1], mode, mask_d2)
        return [d1, d2]
