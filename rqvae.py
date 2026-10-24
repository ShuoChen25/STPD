import json
import os

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


RQVAE_DEFAULTS = {
    "hidden_dim": [256, 128],
    "latent_dim": 128,
    "codebook_sizes": [32, 32],
    "dropout": 0.0,
    "beta": 0.25,
    "epochs": 500,
    "batch_size": 2048,
    "lr": 0.001,
    "log_interval": 10,
    "tag2_align_temp": 0.2,
    "user_tag2_js_weight": 0.0,
    "user_tag2_js_batch_size": 256,
    "codebook_sharing": "private_shared",
}


def build_rqvae_config(args=None, overrides=None):
    cfg = dict(RQVAE_DEFAULTS)
    params = getattr(args, "params", None) if args is not None else None
    key_map = {
        "rqvae_hidden_dim": "hidden_dim",
        "rqvae_latent_dim": "latent_dim",
        "rqvae_codebook_sizes": "codebook_sizes",
        "rqvae_dropout": "dropout",
        "rqvae_beta": "beta",
        "rqvae_epochs": "epochs",
        "rqvae_batch_size": "batch_size",
        "rqvae_lr": "lr",
        "rqvae_log_interval": "log_interval",
        "rqvae_tag2_align_temp": "tag2_align_temp",
        "rqvae_user_tag2_js_weight": "user_tag2_js_weight",
        "rqvae_user_tag2_js_batch_size": "user_tag2_js_batch_size",
        "rqvae_codebook_sharing": "codebook_sharing",
    }
    if isinstance(params, dict):
        for src, dst in key_map.items():
            if src in params and params[src] is not None:
                cfg[dst] = params[src]
    if overrides:
        cfg.update(overrides)
    cfg["log_interval"] = max(1, int(cfg.get("log_interval", 10)))
    cfg["tag2_align_temp"] = max(float(cfg.get("tag2_align_temp", 0.2) or 0.2), 1e-6)
    cfg["user_tag2_js_weight"] = float(cfg.get("user_tag2_js_weight", 0.0) or 0.0)
    cfg["user_tag2_js_batch_size"] = max(1, int(cfg.get("user_tag2_js_batch_size", 256)))
    cfg["codebook_sharing"] = str(
        cfg.get("codebook_sharing", "private_shared")
    ).strip().lower()
    if cfg["codebook_sharing"] not in ("private_shared", "shared_shared"):
        raise ValueError(
            "rqvae_codebook_sharing must be 'private_shared' or "
            f"'shared_shared', got {cfg['codebook_sharing']!r}."
        )
    return cfg


def per_sample_l2_loss(pred, target):
    loss = (pred - target).pow(2.0).reshape(pred.shape[0], -1).sum(dim=1)
    return loss.mean()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super().__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        self.out_norm = nn.LayerNorm(latent_size, elementwise_affine=False)
        hidden_sizes = [input_size] + hidden_sizes
        for in_s, out_s in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.mlp_blocks.append(nn.Sequential(
                nn.Linear(in_s, out_s), nn.LayerNorm(out_s), nn.ReLU(), nn.Dropout(p=dropout),
            ))
            self.residuals.append(
                nn.Conv1d(1, out_s, kernel_size=in_s, bias=False, stride=in_s)
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze(-1)
            x = self.mlp_blocks[i](x) + res
        return self.out_norm(self.out_projection(x))


class VQEmbedding(nn.Embedding):

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99,
                 restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()
        embed_dim = codebook_t.shape[0]
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq, inputs_flat, codebook_t, alpha=-2.0,
        )
        return distances.reshape(*inputs.shape[:-1], -1)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        return self.compute_distances(inputs).argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        return x + torch.rand_like(x) * std

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        n_vectors = vectors.shape[0]

        one_hot = vectors.new_zeros(n_embed, n_vectors)
        one_hot.scatter_(0, idxs.unsqueeze(0), vectors.new_ones(1, n_vectors))
        cluster_size = one_hot.sum(dim=1)
        vectors_sum = one_hot @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            _rand = vectors[torch.randperm(vectors.shape[0], device=vectors.device)][:n_embed]
            if dist.is_initialized():
                dist.broadcast(_rand, 0)
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_rand * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        norm_size = n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        self.weight[:-1, :] = self.embed_ema / norm_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs)
        embeds = self.embed(embed_idxs)
        if self.ema and self.training:
            self._update_embedding()
        return embeds, embed_idxs

    def embed(self, idxs):
        return super().forward(idxs)


class RQBottleneck(nn.Module):

    def __init__(self, latent_shape, code_shape, decay=0.99,
                 shared_codebook=False, restart_unused_codes=True):
        super().__init__()
        self.latent_shape = latent_shape
        self.code_shape = code_shape
        self.n_embed = list(code_shape)
        codebooks = [
            VQEmbedding(self.n_embed[i], latent_shape, decay=decay,
                        restart_unused_codes=restart_unused_codes)
            for i in range(len(code_shape))
        ]
        self.codebooks = nn.ModuleList(codebooks)

    def quantize(self, x):
        residual = x.detach().clone()
        quant_list, code_list = [], []
        aggregated = torch.zeros_like(x)
        for i in range(len(self.code_shape)):
            quant, code = self.codebooks[i](residual)
            residual.sub_(quant)
            aggregated.add_(quant)
            quant_list.append(aggregated.clone())
            code_list.append(code.unsqueeze(-1))
        return quant_list, torch.cat(code_list, dim=-1)

    def forward(self, x):
        quant_list, codes = self.quantize(x)
        loss = torch.mean(torch.stack(
            [per_sample_l2_loss(x, q.detach()) for q in quant_list]
        ))
        quants = x + (quant_list[-1] - x).detach()
        return quants, loss, codes


class RQVAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, codebook_sizes,
                 dropout, latent_loss_weight=0.25):
        super().__init__()
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        rev = list(reversed(hidden_sizes))
        self.decoder = MLP(latent_size, rev, input_size, dropout=dropout)
        self.quantizer = RQBottleneck(
            latent_shape=latent_size,
            code_shape=list(codebook_sizes),
        )
        self.latent_loss_weight = latent_loss_weight

    def forward(self, xs):
        z_e = self.encoder(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decoder(z_q)
        return out, quant_loss, code

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encoder(xs)
        _, _, code = self.quantizer(z_e)
        return code


class JointRQVAE(nn.Module):

    def __init__(self, input_size, hidden_sizes, latent_size, codebook_sizes,
                 dropout, codebook_sharing="private_shared"):
        super().__init__()
        if codebook_sharing not in ("private_shared", "shared_shared"):
            raise ValueError(f"Unsupported codebook sharing: {codebook_sharing!r}")
        self.codebook_sharing = codebook_sharing
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        rev = list(reversed(hidden_sizes))
        self.decoder = MLP(latent_size, rev, input_size, dropout=dropout)
        if codebook_sharing == "shared_shared":
            self.codebook1_shared = VQEmbedding(codebook_sizes[0], latent_size)
        else:
            self.codebook1_d1 = VQEmbedding(codebook_sizes[0], latent_size)
            self.codebook1_d2 = VQEmbedding(codebook_sizes[0], latent_size)
        self.codebook2 = VQEmbedding(codebook_sizes[1], latent_size)

    def level1_codebook(self, domain):
        if self.codebook_sharing == "shared_shared":
            return self.codebook1_shared
        return self.codebook1_d1 if int(domain) == 0 else self.codebook1_d2

    def _tag2_soft_assign(self, z_e_residual, temp):
        codebook = self.codebook2.weight[:-1, :]
        distances = (
            z_e_residual.pow(2.0).sum(dim=1, keepdim=True)
            + codebook.pow(2.0).sum(dim=1).unsqueeze(0)
            - 2.0 * z_e_residual @ codebook.t()
        )
        return F.softmax(-distances / temp, dim=-1)

    def _quantize_level1(self, z_e, domain):
        if self.codebook_sharing == "shared_shared":
            return self.codebook1_shared(z_e)
        q1 = torch.zeros_like(z_e)
        idx1 = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
        m1 = (domain == 0)
        m2 = ~m1
        if m1.any():
            q, i = self.codebook1_d1(z_e[m1])
            q1[m1] = q
            idx1[m1] = i
        if m2.any():
            q, i = self.codebook1_d2(z_e[m2])
            q1[m2] = q
            idx1[m2] = i
        return q1, idx1

    def forward(self, xs, domain):
        z_e = self.encoder(xs)
        q1, idx1 = self._quantize_level1(z_e, domain)
        residual = z_e - q1
        q2, idx2 = self.codebook2(residual)
        quant_loss = 0.5 * (
            per_sample_l2_loss(z_e, q1.detach())
            + per_sample_l2_loss(z_e, (q1 + q2).detach())
        )
        z_q = z_e + (q1 + q2 - z_e).detach()
        out = self.decoder(z_q)
        return out, quant_loss, idx1, idx2

    @torch.no_grad()
    def get_codes(self, xs, domain):
        z_e = self.encoder(xs)
        q1, idx1 = self._quantize_level1(z_e, domain)
        _, idx2 = self.codebook2(z_e - q1)
        return idx1, idx2


def _build_shared_user_histories(args):
    n_shared = args.data["n_shared_users"]
    d1_hist = [[] for _ in range(n_shared)]
    d2_hist = [[] for _ in range(n_shared)]
    for u, i, *_ in args.d1["train"]:
        if u < n_shared:
            d1_hist[u].append(i)
    for u, i, *_ in args.d2["train"]:
        if u < n_shared:
            d2_hist[u].append(i)
    users = [u for u in range(n_shared) if d1_hist[u] and d2_hist[u]]
    return d1_hist, d2_hist, users


def _tag2_interest_distribution(model, data, items, owners, n_users, temp, domain):
    item_t = torch.tensor(items, dtype=torch.long, device=data.device)
    owner_t = torch.tensor(owners, dtype=torch.long, device=data.device)
    z_e = model.encoder(data[item_t])
    cb1 = model.level1_codebook(domain)
    with torch.no_grad():
        idx1 = cb1.find_nearest_embedding(z_e)
        q1 = cb1.embed(idx1)
    residual = z_e - q1
    p_tag2 = model._tag2_soft_assign(residual, temp)
    out = p_tag2.new_zeros(n_users, p_tag2.shape[1])
    out.index_add_(0, owner_t, p_tag2)
    cnt = torch.bincount(owner_t, minlength=n_users).to(p_tag2.dtype).unsqueeze(1)
    return out / cnt.clamp_min(1.0)


def _shared_user_tag2_js_loss(model, d1_data, d2_data, d1_hist, d2_hist,
                              users, batch_size, temp, eps=1e-8):
    n = min(batch_size, len(users))
    picked = torch.randperm(len(users))[:n].tolist()
    picked = [users[i] for i in picked]

    d1_items, d1_owner, d2_items, d2_owner = [], [], [], []
    for row, u in enumerate(picked):
        d1_items.extend(d1_hist[u])
        d1_owner.extend([row] * len(d1_hist[u]))
        d2_items.extend(d2_hist[u])
        d2_owner.extend([row] * len(d2_hist[u]))

    p1 = _tag2_interest_distribution(model, d1_data, d1_items, d1_owner, n, temp, domain=0)
    p2 = _tag2_interest_distribution(model, d2_data, d2_items, d2_owner, n, temp, domain=1)
    p1 = p1.clamp_min(eps)
    p2 = p2.clamp_min(eps)
    p1 = p1 / p1.sum(dim=1, keepdim=True)
    p2 = p2 / p2.sum(dim=1, keepdim=True)
    pm = 0.5 * (p1 + p2)
    js = 0.5 * (p1 * (p1.log() - pm.log())).sum(dim=1)
    js = js + 0.5 * (p2 * (p2.log() - pm.log())).sum(dim=1)
    return js.mean()


def load_raw_item_embeddings(args):
    source_path = os.path.join(
        "LLM", "embedding_output",
        f"{args.d1['name'].lower()}_{args.d2['name'].lower()}_item_title_embeddings.json",
    )
    if not os.path.exists(source_path):
        print(f"Warning: raw embedding file not found: {source_path}")
        return None

    with open(source_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    embedding_key = metadata.get("embedding_key", "TitleEmbedding")
    domains_payload = payload.get("domains", payload if isinstance(payload, dict) else {})

    raw_dim = None

    def _extract_domain(domain_name, n_items):
        nonlocal raw_dim
        domain_data = {}
        if not isinstance(domains_payload, dict):
            return domain_data
        dp = None
        for k, v in domains_payload.items():
            if isinstance(k, str) and k.lower() == domain_name.lower() and isinstance(v, dict):
                dp = v
                break
        if dp is None:
            return domain_data
        for item_id_str, rec in dp.items():
            if not isinstance(rec, dict):
                continue
            try:
                item_id = int(item_id_str)
            except (TypeError, ValueError):
                continue
            if item_id < 0 or item_id >= n_items:
                continue
            vec = rec.get(embedding_key)
            if not isinstance(vec, list):
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                continue
            if raw_dim is None:
                raw_dim = int(arr.size)
            if arr.size != raw_dim:
                continue
            domain_data[item_id] = arr
        return domain_data

    d1_map = _extract_domain(args.d1["name"], args.d1["n_items"])
    d2_map = _extract_domain(args.d2["name"], args.d2["n_items"])

    if raw_dim is None:
        return None

    present_vecs = list(d1_map.values()) + list(d2_map.values())
    if present_vecs:
        norms = np.linalg.norm(np.stack(present_vecs, axis=0), axis=1)
        target_norm = float(norms.mean()) if norms.size > 0 else 1.0
    else:
        target_norm = 1.0
    if not np.isfinite(target_norm) or target_norm <= 0:
        target_norm = 1.0

    def _init_random(n_items, present):
        data = torch.randn(n_items, raw_dim, dtype=torch.float32)
        row_norms = data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        data = data * (target_norm / row_norms)
        for iid, vec in present.items():
            data[iid] = torch.from_numpy(vec)
        return data

    d1_data = _init_random(args.d1["n_items"], d1_map)
    d2_data = _init_random(args.d2["n_items"], d2_map)

    n_d1_missing = args.d1["n_items"] - len(d1_map)
    n_d2_missing = args.d2["n_items"] - len(d2_map)
    print(f"Loaded raw item embeddings: dim={raw_dim}, "
          f"d1={len(d1_map)}/{args.d1['n_items']}, d2={len(d2_map)}/{args.d2['n_items']}, "
          f"random-init missing items: d1={n_d1_missing}, d2={n_d2_missing} "
          f"(target_norm={target_norm:.4f})")
    return {"d1_data": d1_data, "d2_data": d2_data, "raw_dim": raw_dim}


def extract_item_rqcodes(args, cfg_overrides=None):
    from torch.optim import Adam
    from torch.utils.data import TensorDataset, DataLoader

    cfg = build_rqvae_config(args, cfg_overrides)

    raw_bundle = load_raw_item_embeddings(args)
    if raw_bundle is None:
        print("Warning: no raw item embeddings for RQVAE, skipping.")
        return None

    d1_data = raw_bundle["d1_data"]
    d2_data = raw_bundle["d2_data"]
    input_dim = raw_bundle["raw_dim"]
    cs = cfg["codebook_sizes"]

    joint_data = torch.cat([d1_data, d2_data], dim=0).to(args.device)
    joint_domain = torch.cat([
        torch.zeros(d1_data.shape[0], dtype=torch.long),
        torch.ones(d2_data.shape[0], dtype=torch.long),
    ]).to(args.device)
    d1_data_dev = joint_data[:d1_data.shape[0]]
    d2_data_dev = joint_data[d1_data.shape[0]:]
    d1_user_hist, d2_user_hist, shared_users = _build_shared_user_histories(args)

    print(f"=== Joint RQ-VAE: codebook_sharing={cfg['codebook_sharing']} ===")
    if cfg["user_tag2_js_weight"] > 0:
        print(f"Overlap-user Tag2-interest JS alignment: "
              f"weight={cfg['user_tag2_js_weight']}, "
              f"temp={cfg['tag2_align_temp']}, "
              f"batch_users={cfg['user_tag2_js_batch_size']}, "
              f"overlap_users={len(shared_users)}")
    model = JointRQVAE(
        input_dim, cfg["hidden_dim"], cfg["latent_dim"],
        cs, cfg["dropout"], cfg["codebook_sharing"],
    ).to(args.device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(
        TensorDataset(joint_data, joint_domain),
        batch_size=cfg["batch_size"], shuffle=True,
    )
    for ep in range(cfg["epochs"]):
        total_loss = total_recon = total_mse = total_quant = 0.0
        total_user_tag2_js = 0.0
        for batch_x, batch_dom in loader:
            out, quant_loss, _, _ = model(batch_x, batch_dom)
            recon_loss = per_sample_l2_loss(out, batch_x)
            mse_per_dim = F.mse_loss(out, batch_x)
            user_tag2_js_loss = joint_data.new_zeros(())
            if cfg["user_tag2_js_weight"] > 0 and shared_users:
                user_tag2_js_loss = _shared_user_tag2_js_loss(
                    model, d1_data_dev, d2_data_dev,
                    d1_user_hist, d2_user_hist, shared_users,
                    cfg["user_tag2_js_batch_size"], cfg["tag2_align_temp"],
                )
            loss = (
                recon_loss
                + cfg["beta"] * quant_loss
                + cfg["user_tag2_js_weight"] * user_tag2_js_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_mse += mse_per_dim.item()
            total_quant += quant_loss.item()
            total_user_tag2_js += user_tag2_js_loss.item()
        if (ep + 1) % cfg["log_interval"] == 0 or ep == 0 or ep + 1 == cfg["epochs"]:
            d = len(loader)
            print(f"  Epoch {ep+1}/{cfg['epochs']} "
                  f"loss={total_loss/d:.6f} recon_l2={total_recon/d:.6f} "
                  f"mse={total_mse/d:.8f} quant={total_quant/d:.6f} "
                  f"user_tag2_js={total_user_tag2_js/d:.6f}")
    model.eval()

    with torch.no_grad():
        d1_x = d1_data.to(args.device)
        d2_x = d2_data.to(args.device)
        d1_dom = torch.zeros(d1_data.shape[0], dtype=torch.long, device=args.device)
        d2_dom = torch.ones(d2_data.shape[0], dtype=torch.long, device=args.device)
        d1_idx1, d1_idx2 = model.get_codes(
            d1_x, d1_dom,
        )
        d2_idx1, d2_idx2 = model.get_codes(
            d2_x, d2_dom,
        )
        d1_z = model.encoder(d1_x)
        d2_z = model.encoder(d2_x)
        d1_q1, _ = model._quantize_level1(d1_z, d1_dom)
        d2_q1, _ = model._quantize_level1(d2_z, d2_dom)
        d1_tag2_soft = model._tag2_soft_assign(d1_z - d1_q1, cfg["tag2_align_temp"])
        d2_tag2_soft = model._tag2_soft_assign(d2_z - d2_q1, cfg["tag2_align_temp"])

    d1_cb1 = model.level1_codebook(0).weight[:-1, :].detach().cpu().clone()
    d2_cb1 = model.level1_codebook(1).weight[:-1, :].detach().cpu().clone()
    return {
        "codebook_sharing": cfg["codebook_sharing"],
        "d1_item_rqcodes": torch.stack([d1_idx1, d1_idx2], dim=-1).cpu(),
        "d2_item_rqcodes": torch.stack([d2_idx1, d2_idx2], dim=-1).cpu(),
        "d1_item_tag2_soft": d1_tag2_soft.detach().cpu(),
        "d2_item_tag2_soft": d2_tag2_soft.detach().cpu(),
        "d1_codebook1": d1_cb1,
        "d2_codebook1": d2_cb1,
        "shared_codebook2": model.codebook2.weight[:-1, :].detach().cpu().clone(),
    }


def train_rqvae_and_extract_codes(args):
    result = extract_item_rqcodes(args)
    if result is None:
        return

    save_path = os.path.join(args.material_path, "rqvae_codes.pt")
    torch.save(result, save_path)
    print(f"RQVAE codes & codebooks saved to {save_path}")


if __name__ == "__main__":
    import argparse
    import yaml
    from utils import load_data, set_device, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="Movie-Music", help="Domain pair, e.g. Movie-Music / Cell-Elec / Sport-Cloth.", type=str)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    args.dataset = "Amazon"
    args.device = set_device(args.gpu)
    set_seed(args.seed)

    with open("config.yaml", "r") as f:
        args.params = yaml.safe_load(f)

    load_data(args)
    train_rqvae_and_extract_codes(args)
