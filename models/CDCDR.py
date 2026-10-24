
from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import full_rank_eval



def _extract(a, t, x_shape):
    out = a.gather(-1, t.cpu())
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(t.device)


def _linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def _exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    return 1 - torch.exp(-beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))


def _sqrt_beta_schedule(timesteps, max_beta=0.999):
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        f = lambda t: 1 - np.sqrt(t + 0.0001)
        betas.append(min(1 - f(t2) / f(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class _SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def _gather_last(seq, idx):
    expanded = idx.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
    return seq.gather(dim=1, index=expanded).squeeze(1)



class CDCDR(nn.Module):

    def __init__(self, args):
        super().__init__()
        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = int(params.get("embedding_size", 128))
        self.history_len = int(params.get("cdcdr_history_len", 10))
        self.aggregator = params.get("cdcdr_aggregator", "transformer")
        self.gamma = float(params.get("cdcdr_gamma", 0.999))
        self.n_heads = int(params.get("cdcdr_n_heads", 4))
        self.t_layers = int(params.get("cdcdr_n_layers", 1))
        self.dropout_p = float(params.get("cdcdr_dropout", 0.1))
        self.layer_norm = bool(params.get("cdcdr_layer_norm", True))
        self.mean_pooling = bool(params.get("cdcdr_mean_pooling", False))
        self.loss_n = params.get("cdcdr_loss_n", "bpr")

        self.diffuser_type = params.get("cdcdr_diffuser_type", "mlp1")
        self.timesteps = int(params.get("cdcdr_timesteps", 2000))
        self.uncon_w = float(params.get("cdcdr_uncon_w", 5.0))
        self.uncon_p = float(params.get("cdcdr_uncon_p", 0.1))
        self.beta_sche = params.get("cdcdr_beta_sche", "linear")
        self.linespace = int(params.get("cdcdr_linespace", 100))
        self.loss_balance = float(params.get("cdcdr_loss_balance", 0.5))
        self.diffusion_weight = float(params.get("cdcdr_diffusion_weight", 1.0))

        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_emb = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_item_emb = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)
        self.none_embedding = nn.Embedding(1, self.emb_dim, dtype=self.dtype)

        if self.aggregator == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.n_heads,
                dim_feedforward=self.emb_dim,
                dropout=self.dropout_p,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=self.t_layers)
            self.UI_map = nn.Identity()
        else:
            self.UI_map = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            if self.aggregator in ("user_attention", "self_attention"):
                self.W_k = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.Tanh())
                if self.aggregator == "self_attention":
                    self.W_q = nn.Linear(self.emb_dim, 1, bias=False)
        self.ln_1 = nn.LayerNorm(self.emb_dim) if self.layer_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(self.dropout_p)

        self.step_mlp = nn.Sequential(
            _SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        if self.diffuser_type == "mlp1":
            self.diffu_mlp = nn.Sequential(nn.Linear(self.emb_dim * 3, self.emb_dim))
        else:
            self.diffu_mlp = nn.Sequential(
                nn.Linear(self.emb_dim * 3, self.emb_dim * 2),
                nn.GELU(),
                nn.Linear(self.emb_dim * 2, self.emb_dim),
            )

        if self.beta_sche == "linear":
            betas = _linear_beta_schedule(self.timesteps, 0.0001, 0.02)
        elif self.beta_sche == "exp":
            betas = _exp_beta_schedule(self.timesteps)
        elif self.beta_sche == "cosine":
            betas = _cosine_beta_schedule(self.timesteps)
        else:
            betas = _sqrt_beta_schedule(self.timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        idx = list(range(0, self.timesteps + 1, self.linespace))
        self.sub_timesteps = len(idx)
        idx_now = [max(i - 1, 0) for i in idx]
        ac_ddim = alphas_cumprod[idx_now]
        ac_ddim_prev = F.pad(ac_ddim[:-1], (1, 0), value=1.0)
        sqrt_recipm1 = torch.sqrt(1.0 / ac_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(ac_ddim_prev) - torch.sqrt(1.0 - ac_ddim_prev) / sqrt_recipm1
        self.posterior_ddim_coef2 = torch.sqrt(1.0 - ac_ddim_prev) / torch.sqrt(1.0 - ac_ddim)

        d1_hist, d1_mask, d1_len = self._build_history(args.d1["train"], domain=1)
        d2_hist, d2_mask, d2_len = self._build_history(args.d2["train"], domain=2)
        self.register_buffer("d1_history_item", d1_hist)
        self.register_buffer("d1_history_mask", d1_mask)
        self.register_buffer("d1_history_len", d1_len)
        self.register_buffer("d2_history_item", d2_hist)
        self.register_buffer("d2_history_mask", d2_mask)
        self.register_buffer("d2_history_len", d2_len)

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def _to_union(self, raw_users, domain):
        if domain == 1:
            return raw_users
        return torch.where(
            raw_users < self.n_shared_users,
            raw_users,
            raw_users + (self.d1_n_users - self.n_shared_users),
        )

    def _build_history(self, train_rows, domain):
        L = self.history_len
        history = torch.zeros(self.n_users, L, dtype=torch.long)
        mask = torch.zeros(self.n_users, L, dtype=self.dtype)
        lens = torch.zeros(self.n_users, dtype=torch.long)

        u2items = defaultdict(list)
        for row in train_rows:
            u2items[int(row[0])].append(int(row[1]))

        n_sh = self.n_shared_users
        d1_n = self.d1_n_users
        for raw_u, items in u2items.items():
            union_u = raw_u if domain == 1 or raw_u < n_sh else raw_u + (d1_n - n_sh)
            kept = items[-L:]
            k = len(kept)
            history[union_u, :k] = torch.tensor(kept, dtype=torch.long)
            mask[union_u, :k] = 1.0
            lens[union_u] = k
        return history, mask, lens


    def _aggregate(self, user_e, hist_item_e, hist_mask, hist_len):
        hist_item_e = hist_item_e * hist_mask.unsqueeze(-1)

        if self.aggregator == "mean":
            denom = hist_len.unsqueeze(1).to(hist_item_e.dtype) + 1e-10
            out = hist_item_e.sum(dim=1) / denom
        elif self.aggregator in ("user_attention", "self_attention"):
            x = self.ln_1(hist_item_e)
            key = self.W_k(x)
            if self.aggregator == "user_attention":
                attn = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
            else:
                attn = self.W_q(key).squeeze(2)
            attn = attn - attn.max(dim=1, keepdim=True).values
            e = torch.exp(attn) * hist_mask
            w = e / (e.sum(dim=1, keepdim=True) + 1e-10)
            out = torch.matmul(w.unsqueeze(1), hist_item_e).squeeze(1)
        else:
            pad_mask = hist_mask < 0.5
            full_pad = pad_mask.all(dim=1)
            pad_mask = pad_mask.clone()
            pad_mask[full_pad, 0] = False

            x = self.ln_1(hist_item_e)
            x = self.emb_dropout(x)
            seq = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
            if self.mean_pooling:
                m = (~pad_mask).to(seq.dtype).unsqueeze(-1)
                denom = hist_len.unsqueeze(1).to(seq.dtype) + 1e-10
                out = (seq * m).sum(dim=1) / denom
            else:
                valid = (~pad_mask).long()
                last_idx = valid.size(1) - 1 - valid.flip(dims=[1]).argmax(dim=1)
                last_idx = torch.where(valid.sum(dim=1) > 0, last_idx, torch.zeros_like(last_idx))
                out = _gather_last(seq, last_idx)

        out = self.UI_map(out)
        return self.gamma * user_e + (1 - self.gamma) * out

    def _user_condition(self, union_user, domain, pos_item=None):
        u_e = self.user_emb(union_user)
        if domain == 1:
            hist = self.d1_history_item[union_user]
            mask = self.d1_history_mask[union_user]
            hist_len = self.d1_history_len[union_user]
            item_table = self.d1_item_emb
        else:
            hist = self.d2_history_item[union_user]
            mask = self.d2_history_mask[union_user]
            hist_len = self.d2_history_len[union_user]
            item_table = self.d2_item_emb

        if pos_item is not None:
            is_pos = (hist == pos_item.unsqueeze(1)) & (mask > 0.5)
            mask = mask * (~is_pos).to(mask.dtype)
            hist_len = hist_len - is_pos.sum(dim=1)

        item_e = item_table(hist)
        return self._aggregate(u_e, item_e, mask, hist_len)


    def _q_sample(self, x_start, t, noise):
        a = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a * x_start + b * noise

    def _add_uncon(self, h):
        B = h.shape[0]
        m = (torch.sign(torch.rand(B, device=h.device) - self.uncon_p) + 1) / 2
        m = m.view(B, 1)
        none_e = self.none_embedding(torch.zeros(1, dtype=torch.long, device=h.device))
        return h * m + none_e * (1 - m)

    def _denoise(self, x, h, step):
        t = self.step_mlp(step)
        return self.diffu_mlp(torch.cat((x, h, t), dim=1))

    def _denoise_uncon(self, x, step):
        none_e = self.none_embedding(torch.zeros(1, dtype=torch.long, device=x.device))
        h = none_e.expand(x.shape[0], -1)
        return self._denoise(x, h, step)

    @torch.no_grad()
    def _i_sample(self, x, h, t, t_index):
        x_start = (1 + self.uncon_w) * self._denoise(x, h, t) - self.uncon_w * self._denoise_uncon(x, t)
        c1 = self.posterior_ddim_coef1[t_index].to(x.device)
        c2 = self.posterior_ddim_coef2[t_index].to(x.device)
        return c1 * x_start + c2 * x

    @torch.no_grad()
    def _sample_from_noise(self, h):
        x = torch.randn_like(h)
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0],), n * self.linespace, device=h.device, dtype=torch.long)
            x = self._i_sample(x, h, step, n)
        return x


    def _domain_loss(self, inter, neg_item, domain):
        raw_user = inter[:, 0]
        pos_item = inter[:, 1]
        union_user = self._to_union(raw_user, domain)

        cond = self._user_condition(union_user, domain, pos_item=pos_item)

        item_table = self.d1_item_emb if domain == 1 else self.d2_item_emb
        pos_e = item_table(pos_item)
        neg_e = item_table(neg_item)

        pos_score = (cond * pos_e).sum(dim=-1)
        neg_score = (cond * neg_e).sum(dim=-1)

        if self.loss_n == "bce":
            rec_loss = self.bce(pos_score, torch.ones_like(pos_score)) + \
                       self.bce(neg_score, torch.zeros_like(neg_score))
        elif self.loss_n == "bpr":
            rec_loss = -F.logsigmoid(pos_score - neg_score).mean()
        else:
            rec_loss = self.mse(cond, pos_e)

        n = torch.randint(0, self.timesteps, (pos_e.shape[0],), device=pos_e.device).long()
        noise = torch.randn_like(pos_e)
        x_noisy = self._q_sample(pos_e, n, noise)
        h = self._add_uncon(cond)
        predicted = self._denoise(x_noisy, h, n)
        diff_loss = F.mse_loss(pos_e, predicted)

        return rec_loss + self.diffusion_weight * diff_loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        loss_d1 = self._domain_loss(inter_d1, neg_d1_item, domain=1)
        loss_d2 = self._domain_loss(inter_d2, neg_d2_item, domain=2)
        return self.loss_balance * loss_d1 + (1 - self.loss_balance) * loss_d2


    @torch.no_grad()
    def _eval_domain(self, eval_set, domain, history, domain_name):
        item_emb = self.d1_item_emb.weight if domain == 1 else self.d2_item_emb.weight

        def score_fn(users):
            union = self._to_union(users, domain)
            cond = self._user_condition(union, domain)
            x = self._sample_from_noise(cond)
            return x @ item_emb.t()

        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 1, mask_d1, self.d1_domain_name)
        if domain == 2:
            return self._eval_domain(eval_set_2, 2, mask_d2, self.d2_domain_name)
        d1 = self._eval_domain(eval_set_1, 1, mask_d1, self.d1_domain_name)
        d2 = self._eval_domain(eval_set_2, 2, mask_d2, self.d2_domain_name)
        return [d1, d2]
