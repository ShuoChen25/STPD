import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial

from utils import full_rank_eval


# ───────────────────────── Layers ─────────────────────────

class _Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real_value_mask):
        return (real_value_mask >= 0).float().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _create_projection_matrix(m, d):
    nb_full_blocks = max(int(m / d), 1)
    blocks = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        blocks.append(q.T)
    final_matrix = torch.vstack(blocks)
    multiplier = torch.norm(torch.randn((nb_full_blocks * d, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)


def _softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5

    data_dash = torch.einsum("nhd,md->nhm", data_normalizer * data, projection_matrix)
    diag_data = (data ** 2).sum(-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )
    return data_dash


def _kernelized_softmax(query, key, value, projection_matrix, tau=0.5):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_kernel = _softmax_kernel(query, True, projection_matrix)
    key_kernel = _softmax_kernel(key, False, projection_matrix)
    kvs = torch.einsum("nhm,nhd->hmd", key_kernel, value)
    numerator = torch.einsum("nhm,hmd->nhd", query_kernel, kvs)
    denominator = (query_kernel * key_kernel.sum(0, keepdim=True)).sum(-1, keepdim=True)
    return numerator / denominator


class _KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_factor, nb_random_features=64):
        super().__init__()
        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        self.Wv = nn.ModuleList()
        for _ in range(num_factor):
            self.Wk.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
            self.Wq.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
            self.Wv.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
        self.num_factor = num_factor
        self.nb_random_features = nb_random_features

    def forward(self, z, tau):
        query = torch.zeros_like(z)
        key = torch.zeros_like(z)
        value = torch.zeros_like(z)
        for h in range(self.num_factor):
            query[:, h] = self.Wq[h](z[:, h])
            key[:, h] = self.Wk[h](z[:, h])
            value[:, h] = self.Wv[h](z[:, h])
        dim = query.shape[-1]
        proj = _create_projection_matrix(self.nb_random_features, dim).to(query.device)
        z_next = _kernelized_softmax(query, key, value, proj, tau)
        return z_next.flatten(-2, -1)


# ───────────────────────── MFGSLAE model ─────────────────────────

class MFGSLAE(nn.Module):
    """MF-GSLAE adapted to Copenhagen's training/eval pipeline.

    Original paper: Multi-Factor Graph-Structure-Learning AutoEncoder for CDR.
    Domains share a unified user index space:
        d1-only users: [0, d1_n_users), d2 users mapped via shared/non-shared offset.
    Items are concatenated: source items [0, d1_n_items), target items [d1_n_items, n_items).
    """

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        # hyperparameters
        self.lat_dim = int(params.get("mfgslae_lat_dim", 64))
        self.hidden_dim = self.lat_dim
        self.drop_out = float(params.get("mfgslae_dropout", 0.7))
        self.tau = float(params.get("mfgslae_tau", 0.5))
        self.factor = int(params.get("mfgslae_factor", 4))
        self.epsilon = float(params.get("mfgslae_epsilon", 5.0))
        self.alpha = float(params.get("mfgslae_alpha", 0.1))
        self.l1_rate = float(params.get("mfgslae_l1_rate", 1e-6))
        self.ratio = float(params.get("mfgslae_ratio", 0.99))
        self.ratio_threshold = float(params.get("mfgslae_ratio_threshold", 0.5))
        self.nb_random_features = int(params.get("mfgslae_nb_random_features", 64))

        # dataset structure
        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_items = self.d1_n_items + self.d2_n_items

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        # rating matrices (sparse on GPU for encoders, scipy CSR on CPU for loss targets)
        self._build_rating_matrices(args)

        # network
        self.source_encoder = nn.Linear(self.n_items, self.hidden_dim)
        self.target_encoder = nn.Linear(self.n_items, self.hidden_dim)
        self.gsl_encoder = nn.Linear(self.n_items, self.hidden_dim)
        self.act = nn.LeakyReLU()
        self.graph_layer = _KernelAttention(
            self.hidden_dim, self.hidden_dim, self.factor,
            nb_random_features=self.nb_random_features,
        )
        self.subspace_projector = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // self.factor),
                nn.ReLU(),
            ) for _ in range(self.factor)
        )
        self.norm = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        ])
        self.source_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.target_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.proj_source = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.proj_target = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.mask_source = nn.Parameter(torch.empty(1, self.factor))
        self.mask_target = nn.Parameter(torch.empty(1, self.factor))
        self.decoder_source = nn.Linear(self.lat_dim, self.d1_n_items)
        self.decoder_target = nn.Linear(self.lat_dim, self.d2_n_items)
        self.binary = _Binarize.apply

        # init
        self.apply(self._init_weights)
        nn.init.normal_(self.mask_source, 0, 0.01)
        nn.init.normal_(self.mask_target, 0, 0.01)

        # epoch tracker (compatible with main.py: args.model._epoch[0] = i)
        self.register_buffer("_epoch", torch.zeros(1, dtype=torch.long))
        self._last_synced_epoch = -1

    # ---- Initialization ----

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    # ---- Rating matrix construction ----

    def _map_d2_to_unified(self, d2_users):
        return torch.where(
            d2_users < self.n_shared_users,
            d2_users,
            d2_users + (self.d1_n_users - self.n_shared_users),
        )

    def _build_rating_matrices(self, args):
        n_users = self.n_users
        d1_n_items = self.d1_n_items
        n_items = self.n_items
        n_shared = self.n_shared_users
        d1_n_users = self.d1_n_users

        d1_train = np.asarray(args.d1["train"], dtype=np.int64)
        d2_train = np.asarray(args.d2["train"], dtype=np.int64)

        # source: d1 user-id directly equals unified user-id
        src_rows = d1_train[:, 0]
        src_cols = d1_train[:, 1]  # in [0, d1_n_items)

        # target: map d2 user to unified id; offset item ids into combined space
        d2_users = d2_train[:, 0]
        d2_unified = np.where(d2_users < n_shared, d2_users, d2_users + (d1_n_users - n_shared))
        tgt_rows = d2_unified
        tgt_cols_full = d2_train[:, 1] + d1_n_items  # in [d1_n_items, n_items)

        src_input = sp.coo_matrix(
            (np.ones(len(src_rows), dtype=np.float32), (src_rows, src_cols)),
            shape=(n_users, n_items),
        )
        tgt_input = sp.coo_matrix(
            (np.ones(len(tgt_rows), dtype=np.float32), (tgt_rows, tgt_cols_full)),
            shape=(n_users, n_items),
        )
        full_input = (src_input + tgt_input).tocoo()

        # row-L2 normalize then convert to torch sparse on GPU
        self.sparse_norm_rating_source = self._row_norm_to_torch_sparse(src_input)
        self.sparse_norm_rating_target = self._row_norm_to_torch_sparse(tgt_input)
        self.sparse_norm_rating_matrix = self._row_norm_to_torch_sparse(full_input)

        # CPU CSR for batched loss targets (per-domain item space)
        self.rating_matrix_source = sp.coo_matrix(
            (np.ones(len(src_rows), dtype=np.float32), (src_rows, src_cols)),
            shape=(n_users, d1_n_items),
        ).tocsr()
        self.rating_matrix_target = sp.coo_matrix(
            (np.ones(len(tgt_rows), dtype=np.float32), (tgt_rows, d2_train[:, 1])),
            shape=(n_users, self.d2_n_items),
        ).tocsr()

    def _row_norm_to_torch_sparse(self, mat_coo):
        csr = mat_coo.tocsr().astype(np.float32)
        sq_sum = np.asarray(csr.multiply(csr).sum(axis=1)).flatten()
        norms = np.sqrt(sq_sum)
        norms[norms == 0] = 1.0
        normed = sp.diags(1.0 / norms).dot(csr).tocoo()
        indices = torch.LongTensor(np.stack([normed.row, normed.col]))
        values = torch.tensor(normed.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, normed.shape
        ).coalesce().to(self.device)

    # ---- Epoch hook ----

    def _maybe_epoch_start(self):
        if not self.training:
            return
        cur = int(self._epoch[0].item())
        if cur != self._last_synced_epoch:
            if self.ratio > self.ratio_threshold:
                self.ratio = self.ratio * self.ratio
            else:
                self.ratio = self.ratio_threshold
            self._last_synced_epoch = cur

    # ---- Core blocks ----

    def _sparse_drop(self, A):
        if self.training and self.drop_out > 0:
            dist = Binomial(probs=1 - self.drop_out)
            mask = dist.sample(A.values().size()).bool()
            return torch.sparse_coo_tensor(
                A.indices()[:, mask],
                A.values()[mask] / (1 - self.drop_out),
                size=A.size(),
            ).to(A.device)
        return A

    def _factor_division(self, h):
        return torch.stack(
            [proj(h) for proj in self.subspace_projector], dim=1
        )

    def _factor_gsl(self, h):
        h = self.norm[0](h)
        self._h_factor = self._factor_division(h)
        h_enh = self.graph_layer(self._h_factor, self.tau)
        h = h + h_enh
        h = self.act(h)
        h = self.norm[1](h)
        return h

    def _factor_selection(self, h):
        h_src = self.proj_source(h).reshape(h.shape[0], self.factor, -1)
        h_tgt = self.proj_target(h).reshape(h.shape[0], self.factor, -1)
        m_src = self.binary(self.mask_source).unsqueeze(-1)
        m_tgt = self.binary(self.mask_target).unsqueeze(-1)
        h_src_aug = self.source_aug_norm((h_src * m_src).flatten(-2, -1))
        h_tgt_aug = self.target_aug_norm((h_tgt * m_tgt).flatten(-2, -1))
        return h_src_aug, h_tgt_aug

    def _compactness(self, emb):
        emb = F.normalize(emb, dim=-1)
        emb_all = torch.flatten(emb, 0, 1)
        NK, D = emb_all.shape
        loss = -0.5 * torch.logdet(
            torch.eye(D, device=self.device) + (D / NK / self.epsilon * emb_all.T) @ emb_all
        )
        N, K, D = emb.shape
        emb_t = torch.einsum("abc->bca", emb)
        loss_factor = torch.einsum("abc,adc->abd", D / N / self.epsilon * emb_t, emb_t)
        loss_factor = 0.5 * torch.logdet(torch.eye(D, device=self.device).unsqueeze(0) + loss_factor)
        return loss + loss_factor.sum()

    # ---- Encoders / forward ----

    def _encode(self):
        self._maybe_epoch_start()

        src_in = self._sparse_drop(self.sparse_norm_rating_source)
        h_source = self.act(
            torch.sparse.mm(src_in, self.source_encoder.weight.T) + self.source_encoder.bias
        )

        tgt_in = self._sparse_drop(self.sparse_norm_rating_target)
        h_target = self.act(
            torch.sparse.mm(tgt_in, self.target_encoder.weight.T) + self.target_encoder.bias
        )

        gsl_in = self._sparse_drop(self.sparse_norm_rating_matrix)
        h = torch.sparse.mm(gsl_in, self.gsl_encoder.weight.T) + self.gsl_encoder.bias
        h_gsl = self._factor_gsl(self.act(h))
        self._h_gsl = h_gsl.reshape(h_gsl.shape[0], self.factor, -1)

        if self.factor > 1:
            h_src_aug, h_tgt_aug = self._factor_selection(h_gsl)
        else:
            h_src_aug = self.proj_source(h_gsl)
            h_tgt_aug = self.proj_target(h_gsl)

        h_source = self.ratio * h_source + (1 - self.ratio) * h_src_aug
        h_target = self.ratio * h_target + (1 - self.ratio) * h_tgt_aug
        return h_source, h_target

    def forward(self, source_users=None, target_users=None):
        h_source, h_target = self._encode()
        recon_src = None
        recon_tgt = None
        if source_users is not None:
            recon_src = (
                h_source[source_users] @ self.decoder_source.weight.T + self.decoder_source.bias
            )
        if target_users is not None:
            recon_tgt = (
                h_target[target_users] @ self.decoder_target.weight.T + self.decoder_target.bias
            )
        return recon_src, recon_tgt

    # ---- Loss ----

    def _gather_rating_rows(self, csr_mat, user_ids):
        rows = csr_mat[user_ids.detach().cpu().numpy()].toarray()
        return torch.from_numpy(rows).to(self.device, dtype=self.dtype)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        # MF-GSLAE assumes user-level batching: each unique user contributes once.
        # Copenhagen's loop feeds interaction-level batches, so we dedup users here
        # to recover the original loss semantics (and avoid popularity-biased gradients).
        d1_users = torch.unique(inter_d1[:, 0].long())
        d2_users_raw = torch.unique(inter_d2[:, 0].long())
        d2_users = self._map_d2_to_unified(d2_users_raw)

        rating_src = self._gather_rating_rows(self.rating_matrix_source, d1_users)
        rating_tgt = self._gather_rating_rows(self.rating_matrix_target, d2_users)

        recon_src, recon_tgt = self.forward(d1_users, d2_users)

        ce_src = -(F.log_softmax(recon_src, dim=1) * rating_src).sum(1).mean()
        ce_tgt = -(F.log_softmax(recon_tgt, dim=1) * rating_tgt).sum(1).mean()

        if self.factor > 1:
            compact = self.alpha * (
                self._compactness(self._h_factor) + self._compactness(self._h_gsl)
            )
        else:
            compact = torch.zeros((), device=self.device)

        l1 = self.l1_rate * (
            F.l1_loss(self.mask_source, torch.zeros_like(self.mask_source))
            + F.l1_loss(self.mask_target, torch.zeros_like(self.mask_target))
        )
        return ce_src + ce_tgt + compact + l1

    # ---- Evaluation ----

    def _eval_domain(self, eval_set, h, decoder, map_users, domain_name, history):
        """Full in-domain ranking with user-history masking."""
        def score_fn(users):
            unified = map_users(users.long())
            return h[unified] @ decoder.weight.T + decoder.bias
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        h_source, h_target = self._encode()
        if domain == 1:
            return self._eval_domain(
                eval_set_1, h_source, self.decoder_source,
                lambda u: u, self.d1_domain_name, mask_d1,
            )
        if domain == 2:
            return self._eval_domain(
                eval_set_2, h_target, self.decoder_target,
                self._map_d2_to_unified, self.d2_domain_name, mask_d2,
            )
        d1 = self._eval_domain(
            eval_set_1, h_source, self.decoder_source,
            lambda u: u, self.d1_domain_name, mask_d1,
        )
        d2 = self._eval_domain(
            eval_set_2, h_target, self.decoder_target,
            self._map_d2_to_unified, self.d2_domain_name, mask_d2,
        )
        return [d1, d2]
