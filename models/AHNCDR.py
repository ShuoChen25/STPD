import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from utils import full_rank_eval




class _UserEmb(nn.Module):
    def __init__(self, n_users, dim):
        super().__init__()
        self.shared = nn.Embedding(n_users, dim)
        self.t1 = nn.Linear(dim, dim)
        self.t2 = nn.Linear(dim, dim)

    def forward(self, u):
        h = self.shared(u)
        return self.t1(h), self.t2(h)


class _GraphSAGE(nn.Module):
    def __init__(self, n_items, dim, pad_id, neighbor_size=4):
        super().__init__()
        self.emb = nn.Embedding(n_items, dim)
        self.k = neighbor_size
        self.pad_id = pad_id
        self.lin = nn.Linear(dim, dim)
        self.self_lin = nn.Linear(dim, dim)

    def forward(self, seq, mask):
        seq_emb = self.emb(seq)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg = (seq_emb * m).sum(dim=1, keepdim=True) / denom
        agg = agg.expand_as(seq_emb)
        return F.relu(self.self_lin(seq_emb) + self.lin(agg))


class _IUGraphLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.i_to_u = nn.Linear(dim, dim)

    def forward(self, user_feat, seq_feat, mask):
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        seq_mess = (self.i_to_u(seq_feat) * m).sum(dim=1) / denom
        return user_feat + seq_mess


class _RefineIU(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.trans = nn.Linear(dim, dim)

    def forward(self, user_feat, seq_feat, mask):
        kpm = (mask == 0)
        no_valid = kpm.all(dim=1)
        kpm = kpm.clone()
        kpm[no_valid] = False
        seq_feat, _ = self.attn(seq_feat, seq_feat, seq_feat, key_padding_mask=kpm)
        seq_feat = seq_feat * mask.unsqueeze(-1)

        sf = seq_feat.reshape(-1, seq_feat.size(-1))
        att = torch.matmul(user_feat, sf.T)
        flat_mask = mask.reshape(-1)
        att = att.masked_fill(flat_mask.unsqueeze(0) == 0, 0.0)
        att = F.normalize(att, p=1, dim=1)
        agg = torch.matmul(att, sf) / sf.size(0)
        return user_feat + self.trans(agg)


class _SpecificGraphModule(nn.Module):
    def __init__(self, dim, layer_nums, n_items, pad_id):
        super().__init__()
        self.layer_nums = layer_nums
        self.pad_id = pad_id
        self.gnn = _GraphSAGE(n_items, dim, pad_id)
        self.iu = nn.ModuleList([_IUGraphLayer(dim) for _ in range(layer_nums)])
        self.refine = nn.ModuleList([_RefineIU(dim) for _ in range(layer_nums)])

    def forward(self, user_feat, seq):
        mask = (seq != self.pad_id).to(user_feat.dtype)
        seq_feat = self.gnn(seq, mask)
        for i in range(self.layer_nums):
            user_feat = F.relu(self.iu[i](user_feat, seq_feat, mask))
            user_feat = F.relu(self.refine[i](user_feat, seq_feat, mask))
        return user_feat, seq_feat


class _GateFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t1 = nn.Linear(dim, dim)
        self.t2 = nn.Linear(dim, dim)

    def forward(self, msg, slf):
        gate = torch.sigmoid(self.t1(msg) + self.t2(slf))
        return gate * msg + (1 - gate) * slf


class _InterMatchLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tr1 = nn.Linear(dim, dim)
        self.tr2 = nn.Linear(dim, dim)
        self.s1 = nn.Linear(dim, dim)
        self.s2 = nn.Linear(dim, dim)
        self.g1 = _GateFuse(dim)
        self.g2 = _GateFuse(dim)

    def forward(self, u1, u2):
        B = u1.size(0)

        att = torch.matmul(u1, u2.T)
        att = att - torch.diag_embed(torch.diag(att))
        att = F.normalize(att, p=1, dim=1)
        msg = self.tr1(torch.matmul(att, u2) / B)
        sumg = self.g1(msg, self.s1(u2))
        u1n = F.normalize(torch.matmul(torch.matmul(u1, u1.T), u1), p=1, dim=1) + sumg

        att = torch.matmul(u2, u1.T)
        att = att - torch.diag_embed(torch.diag(att))
        att = F.normalize(att, p=1, dim=1)
        msg = self.tr2(torch.matmul(att, u1) / B)
        sumg = self.g2(msg, self.s2(u1))
        u2n = F.normalize(torch.matmul(torch.matmul(u2, u2.T), u2), p=1, dim=1) + sumg

        return u1n, u2n


class _InterMatchModule(nn.Module):
    def __init__(self, dim, layer_nums):
        super().__init__()
        self.layers = nn.ModuleList([_InterMatchLayer(dim) for _ in range(layer_nums)])

    def forward(self, u1, u2):
        for l in self.layers:
            u1, u2 = l(u1, u2)
            u1 = F.relu(u1)
            u2 = F.relu(u2)
        return u1, u2


class _Predict(nn.Module):
    def __init__(self, dim, hid):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim * 2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, u1, u2, i_feat):
        u1e = u1.unsqueeze(1).expand_as(i_feat)
        u2e = u2.unsqueeze(1).expand_as(i_feat)
        l1 = torch.sigmoid(self.fc(torch.cat((u1e, i_feat), -1))).squeeze(-1)
        l2 = torch.sigmoid(self.fc(torch.cat((u2e, i_feat), -1))).squeeze(-1)
        return l1, l2

    def score_one(self, u, i_feat):
        ue = u.unsqueeze(1).expand_as(i_feat)
        return torch.sigmoid(self.fc(torch.cat((ue, i_feat), -1))).squeeze(-1)




class AHNCDR(nn.Module):

    def __init__(self, args):
        super().__init__()

        p = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = p["embedding_size"]
        self.hid_dim = p.get("ahncdr_hid_dim", 32)
        self.seq_len = p.get("ahncdr_seq_len", 20)
        self.m1_layers = p.get("ahncdr_m1_layers", 2)
        self.m3_layers = p.get("ahncdr_m3_layers", 3)
        self.alpha_l = p.get("ahncdr_alpha_l", 3)
        self.proto_reg = p.get("ahncdr_proto_reg", 1e-3)
        self.dbscan_eps = p.get("ahncdr_dbscan_eps", 7.9)
        self.dbscan_min_samples = p.get("ahncdr_dbscan_min_samples", 10)
        self.proc_bs = p.get("ahncdr_proc_bs", 256)
        self.item_chunk = p.get("ahncdr_eval_item_chunk", 512)

        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.n_items_total = self.d1_n_items + self.d2_n_items + 1
        self.pad_id = self.d1_n_items + self.d2_n_items

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.user_emb = _UserEmb(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items_total, self.emb_dim)

        self.gnn_d1 = _SpecificGraphModule(self.emb_dim, self.m1_layers, self.n_items_total, self.pad_id)
        self.gnn_d2 = _SpecificGraphModule(self.emb_dim, self.m1_layers, self.n_items_total, self.pad_id)
        self.inter_match = _InterMatchModule(self.emb_dim, self.m3_layers)

        self.w1 = nn.Parameter(torch.rand(self.emb_dim, self.emb_dim))
        self.w2 = nn.Parameter(torch.rand(self.emb_dim, self.emb_dim))

        self.head = _Predict(self.emb_dim, self.hid_dim)
        self.head_m0 = _Predict(self.emb_dim, self.hid_dim)
        self.head_m1 = _Predict(self.emb_dim, self.hid_dim)
        self.head_m2 = _Predict(self.emb_dim, self.hid_dim)

        self._build_user_seqs(args)

        self._init_clusters()

        self.bce = nn.BCELoss(reduction="none")
        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _d2_user_to_mixed(self, u):
        return u if u < self.n_shared_users else u + self.d1_n_users - self.n_shared_users

    def _pad(self, items):
        items = items[-self.seq_len:]
        if len(items) < self.seq_len:
            items = [self.pad_id] * (self.seq_len - len(items)) + items
        return items

    def _build_user_seqs(self, args):
        d1_hist = defaultdict(list)
        for row in args.d1["train"]:
            u, i = int(row[0]), int(row[1])
            d1_hist[u].append(i)

        d2_hist = defaultdict(list)
        for row in args.d2["train"]:
            u, i = int(row[0]), int(row[1])
            u_mix = self._d2_user_to_mixed(u)
            d2_hist[u_mix].append(i + self.d1_n_items)

        seq_d1 = np.full((self.n_users, self.seq_len), self.pad_id, dtype=np.int64)
        seq_d2 = np.full((self.n_users, self.seq_len), self.pad_id, dtype=np.int64)
        for u, items in d1_hist.items():
            seq_d1[u] = self._pad(items)
        for u, items in d2_hist.items():
            seq_d2[u] = self._pad(items)

        self.register_buffer("seq_d1", torch.from_numpy(seq_d1))
        self.register_buffer("seq_d2", torch.from_numpy(seq_d2))

        overlap = np.zeros(self.n_users, dtype=np.int64)
        overlap[: self.n_shared_users] = 1
        self.register_buffer("user_overlap", torch.from_numpy(overlap))

    def _run_dbscan(self, x_np):
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(x_np)
        unique = np.unique(labels[labels != -1])
        if unique.size == 0:
            cents = x_np.mean(axis=0, keepdims=True)
        else:
            cents = np.stack([x_np[labels == lb].mean(axis=0) for lb in unique], axis=0)

        x_norm = np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-12
        c_norm = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
        sim = 1 - (x_np @ cents.T) / (x_norm * c_norm.T)
        return torch.tensor(cents, dtype=self.dtype), torch.tensor(sim, dtype=self.dtype)

    def _init_clusters(self):
        with torch.no_grad():
            u_all = torch.arange(self.n_users)
            u1, u2 = self.user_emb(u_all)
        cent1, dist1 = self._run_dbscan(u1.detach().cpu().numpy())
        cent2, dist2 = self._run_dbscan(u2.detach().cpu().numpy())
        self.register_buffer("centroids_d1", cent1)
        self.register_buffer("centroids_d2", cent2)
        self.register_buffer("dist_d1", dist1)
        self.register_buffer("dist_d2", dist2)


    def _gather_seqs(self, users):
        return self.seq_d1[users], self.seq_d2[users]

    def _forward_users(self, users):
        u1, u2 = self.user_emb(users)
        seq1, seq2 = self._gather_seqs(users)
        u_m1_d1, _ = self.gnn_d1(u1, seq1)
        u_m1_d2, _ = self.gnn_d2(u2, seq2)
        u_m3_d1, u_m3_d2 = self.inter_match(u_m1_d1, u_m1_d2)
        u_d1 = torch.matmul(u_m3_d1, self.w1) + torch.matmul(u_m3_d2, 1 - self.w1)
        u_d2 = torch.matmul(u_m3_d2, self.w2) + torch.matmul(u_m3_d1, 1 - self.w2)
        return u1, u2, u_m1_d1, u_m1_d2, u_m3_d1, u_m3_d2, u_d1, u_d2

    def _logits(self, head, u1, u2, items):
        i_feat = self.item_emb(items)
        return head(u1, u2, i_feat)

    def _proto_nce(self, u_feat, batch_users, dist_buf, centroids_other):
        if u_feat.size(0) == 0:
            return torch.tensor(0.0, device=u_feat.device)
        K = min(dist_buf.size(1), centroids_other.size(0))
        dist = dist_buf[batch_users][:, :K]
        cents = centroids_other[:K]
        u_norm = F.normalize(u_feat)
        u_pos = torch.matmul(dist, cents)
        pos = torch.exp((u_norm * u_pos).sum(dim=1) / 0.1)
        ttl = torch.exp(torch.matmul(u_norm, u_pos.T) / 0.1).sum(dim=1)
        return self.proto_reg * (-torch.log(pos / (ttl + 1e-12))).mean()


    def _chunk_loss(self, users, pos_items, neg_items, domain_id):
        u1, u2, u_m1_d1, u_m1_d2, u_m3_d1, u_m3_d2, u_d1, u_d2 = self._forward_users(users)
        cand = torch.stack([pos_items, neg_items], dim=1)
        labels = torch.tensor([1.0, 0.0], device=users.device).expand(len(users), 2)

        l_d1, l_d2 = self._logits(self.head, u_d1, u_d2, cand)
        l1_d1, l1_d2 = self._logits(self.head_m1, u_m1_d1, u_m1_d2, cand)
        l2_d1, l2_d2 = self._logits(self.head_m2, u_m3_d1, u_m3_d2, cand)

        mask_d1 = (domain_id == 0).float().unsqueeze(1)
        mask_d2 = (domain_id == 1).float().unsqueeze(1)

        def _balanced(pred1, pred2):
            ld1 = self.bce(pred1, labels) * mask_d1
            ld2 = self.bce(pred2, labels) * mask_d2
            l1 = ld1.mean()
            l2 = ld2.mean()
            r1 = 1 - l1 / (l1 + l2 + 1e-8)
            r2 = 1 - l2 / (l1 + l2 + 1e-8)
            return (ld1 * (1 + r1) + ld2 * (1 + r2)).mean()

        loss = _balanced(l_d1, l_d2) + _balanced(l1_d1, l1_d2) + _balanced(l2_d1, l2_d2)

        overlap = self.user_overlap[users].bool()
        d1_idx = torch.nonzero(overlap & (domain_id == 0), as_tuple=False).squeeze(-1)
        d2_idx = torch.nonzero(overlap & (domain_id == 1), as_tuple=False).squeeze(-1)
        if d1_idx.numel() > 0:
            loss = loss + self._proto_nce(u_m1_d1[d1_idx], users[d1_idx], self.dist_d1, self.centroids_d2)
        if d2_idx.numel() > 0:
            loss = loss + self._proto_nce(u_m1_d2[d2_idx], users[d2_idx], self.dist_d2, self.centroids_d1)
        return loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_users = inter_d1[:, 0].long()
        d1_items = inter_d1[:, 1].long()
        d1_neg = neg_d1_item.long()

        d2_users_raw = inter_d2[:, 0].long()
        d2_users = torch.where(
            d2_users_raw < self.n_shared_users,
            d2_users_raw,
            d2_users_raw + (self.d1_n_users - self.n_shared_users),
        )
        d2_items = inter_d2[:, 1].long() + self.d1_n_items
        d2_neg = neg_d2_item.long() + self.d1_n_items

        users = torch.cat([d1_users, d2_users])
        pos_items = torch.cat([d1_items, d2_items])
        neg_items = torch.cat([d1_neg, d2_neg])
        domain_id = torch.cat([
            torch.zeros(len(d1_users), device=users.device),
            torch.ones(len(d2_users), device=users.device),
        ])

        n = users.size(0)
        perm = torch.randperm(n, device=users.device)
        total = torch.tensor(0.0, device=users.device)
        n_chunks = 0
        for start in range(0, n, self.proc_bs):
            idx = perm[start:start + self.proc_bs]
            total = total + self._chunk_loss(
                users[idx], pos_items[idx], neg_items[idx], domain_id[idx]
            )
            n_chunks += 1
        return total / max(n_chunks, 1)


    def _remap_eval_users(self, users_raw, domain):
        if domain == 1:
            return users_raw
        return torch.where(
            users_raw < self.n_shared_users,
            users_raw,
            users_raw + (self.d1_n_users - self.n_shared_users),
        )

    @torch.no_grad()
    def _eval_domain(self, eval_set, domain, domain_name, history):
        if domain == 1:
            n_items = self.d1_n_items
            item_offset = 0
        else:
            n_items = self.d2_n_items
            item_offset = self.d1_n_items
        device = next(self.parameters()).device
        all_items = torch.arange(item_offset, item_offset + n_items, device=device, dtype=torch.long)

        def score_fn(users_raw):
            users_raw = users_raw.long()
            n_all = users_raw.size(0)
            scores = torch.empty(n_all, n_items, device=device)
            for us in range(0, n_all, self.proc_bs):
                ue = min(n_all, us + self.proc_bs)
                users = self._remap_eval_users(users_raw[us:ue], domain)
                _, _, _, _, _, _, u_d1, u_d2 = self._forward_users(users)
                u = u_d1 if domain == 1 else u_d2
                b = users.size(0)
                for s in range(0, n_items, self.item_chunk):
                    e = min(n_items, s + self.item_chunk)
                    chunk_items = all_items[s:e].unsqueeze(0).expand(b, -1)
                    i_feat = self.item_emb(chunk_items)
                    scores[us:ue, s:e] = self.head.score_one(u, i_feat)
            return scores

        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 1, self.d1_domain_name, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, 2, self.d2_domain_name, mask_d2)
        d1 = self._eval_domain(eval_set_1, 1, self.d1_domain_name, mask_d1)
        d2 = self._eval_domain(eval_set_2, 2, self.d2_domain_name, mask_d2)
        return [d1, d2]
