
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import full_rank_eval



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _get_named_beta_schedule(name, n_steps):
    if name == "linear":
        scale = 1000.0 / n_steps
        return np.linspace(scale * 0.0001, scale * 0.02, n_steps, dtype=np.float64)
    if name == "trunc_lin":
        scale = 1000.0 / n_steps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)
    if name == "cosine":
        return _betas_for_alpha_bar(
            n_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    if name == "sqrt":
        return _betas_for_alpha_bar(n_steps, lambda t: 1 - np.sqrt(t + 0.0001))
    if name == "trunc_cos":
        return _betas_for_alpha_bar_left(
            n_steps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2
        )
    raise NotImplementedError(f"unknown beta schedule: {name}")


def _betas_for_alpha_bar(n_steps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(n_steps):
        t1 = i / n_steps
        t2 = (i + 1) / n_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _betas_for_alpha_bar_left(n_steps, alpha_bar, max_beta=0.999):
    betas = [min(1 - alpha_bar(0), max_beta)]
    for i in range(n_steps - 1):
        t1 = i / n_steps
        t2 = (i + 1) / n_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



class _SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class _LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class _SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = _LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class _PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))
        ))
        return self.w_2(self.dropout(activation))


class _MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [
            l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for l, x in zip(self.linear_layers, (q, k, v))
        ]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(
            hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head)
        )
        return hidden


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super().__init__()
        self.attention = _MultiHeadedAttention(attn_heads, hidden_size, dropout)
        self.feed_forward = _PositionwiseFeedForward(hidden_size, dropout)
        self.input_sublayer = _SublayerConnection(hidden_size, dropout)
        self.output_sublayer = _SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda h: self.attention(h, h, h, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class _TransformerRep(nn.Module):
    def __init__(self, hidden_size, n_blocks, n_heads, dropout):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(hidden_size, n_heads, dropout) for _ in range(n_blocks)]
        )

    def forward(self, hidden, mask):
        for blk in self.transformer_blocks:
            hidden = blk(hidden, mask)
        return hidden


class _DiffuXstart(nn.Module):

    def __init__(self, hidden_size, n_blocks, n_heads, dropout, lambda_uncertainty):
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_uncertainty = lambda_uncertainty
        time_embed_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim), _SiLU(), nn.Linear(time_embed_dim, hidden_size)
        )
        self.att = _TransformerRep(hidden_size, n_blocks, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_diffu_rep = _LayerNorm(hidden_size)

    @staticmethod
    def _timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self._timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        lam = torch.normal(
            mean=torch.full(rep_item.shape, self.lambda_uncertainty),
            std=torch.full(rep_item.shape, self.lambda_uncertainty),
        ).to(x_t.device)
        rep_diffu = self.att(rep_item + lam * x_t.unsqueeze(1), mask_seq)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        return rep_diffu[:, -1, :], rep_diffu


class _DiffuRec(nn.Module):

    def __init__(self, hidden_size, n_blocks, n_heads, dropout, diffusion_steps,
                 noise_schedule, w, lambda_uncertainty, rescale_timesteps):
        super().__init__()
        self.hidden_size = hidden_size
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.w = w
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(_get_named_beta_schedule(noise_schedule, diffusion_steps), dtype=np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.num_timesteps = int(self.betas.shape[0])

        self.xstart_model = _DiffuXstart(
            hidden_size, n_blocks, n_heads, dropout, lambda_uncertainty
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _q_posterior_mean(self, x_start, x_t, t):
        return (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

    def _p_mean_variance(self, item_rep, item_rep1, x_t, t, mask_seq):
        out, _ = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        out1, _ = self.xstart_model(item_rep1, x_t, self._scale_timesteps(t), mask_seq)
        x_0 = (1 + self.w) * out - self.w * out1
        log_var = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        log_var = _extract_into_tensor(log_var, t, x_t.shape)
        mean = self._q_posterior_mean(x_0, x_t, t)
        return mean, log_var

    def _p_sample(self, item_rep, item_rep1, x_t, t, mask_seq):
        mean, log_var = self._p_mean_variance(item_rep, item_rep1, x_t, t, mask_seq)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def reverse_p_sample(self, item_rep, item_rep1, noise_x_t, mask_seq):
        device = next(self.xstart_model.parameters()).device
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * item_rep.shape[0], device=device)
            noise_x_t = self._p_sample(item_rep, item_rep1, noise_x_t, t, mask_seq)
        return noise_x_t

    def forward(self, item_rep, item_tag, mask_seq):
        noise = torch.randn_like(item_tag)
        t = torch.randint(0, self.num_timesteps, (item_rep.shape[0],), device=item_rep.device)
        x_t = self.q_sample(item_tag, t, noise)
        x_0, _ = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        return x_0



class D2TCDR(nn.Module):

    def __init__(self, args):
        super().__init__()
        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = int(params.get("embedding_size", 128))
        self.max_len = int(params.get("d2tcdr_max_len", 8))
        self.n_blocks = int(params.get("d2tcdr_num_blocks", 4))
        self.n_heads = int(params.get("d2tcdr_n_heads", 4))
        self.dropout_p = float(params.get("d2tcdr_dropout", 0.1))
        self.emb_dropout_p = float(params.get("d2tcdr_emb_dropout", 0.3))
        self.diffusion_steps = int(params.get("d2tcdr_diffusion_steps", 32))
        self.noise_schedule = params.get("d2tcdr_noise_schedule", "trunc_lin")
        self.cfg_w = float(params.get("d2tcdr_w", 1.0))
        self.lambda_uncertainty = float(params.get("d2tcdr_lambda_uncertainty", 0.001))
        self.rescale_timesteps = bool(params.get("d2tcdr_rescale_timesteps", True))
        self.loss_lambda = float(params.get("d2tcdr_loss_lambda", 1.0))
        self.lambda_shared = float(params.get("d2tcdr_lambda_shared", 200.0))
        self.lambda_private = float(params.get("d2tcdr_lambda_private", 0.2))
        self.lambda_orthogonal = float(params.get("d2tcdr_lambda_orthogonal", 20.0))

        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.n_shared_users = args.data["n_shared_users"]
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.d1_item_emb = nn.Embedding(self.d1_n_items + 1, self.emb_dim, padding_idx=0)
        self.d2_item_emb = nn.Embedding(self.d2_n_items + 1, self.emb_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_len, self.emb_dim)
        self.shared_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.specific_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.embed_dropout = nn.Dropout(self.emb_dropout_p)
        self.LayerNorm = _LayerNorm(self.emb_dim, eps=1e-12)

        self.diffu = _DiffuRec(
            hidden_size=self.emb_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout_p,
            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
            w=self.cfg_w,
            lambda_uncertainty=self.lambda_uncertainty,
            rescale_timesteps=self.rescale_timesteps,
        )

        d1_hist, d1_mask = self._build_history(args.d1["train"], self.d1_n_users)
        d2_hist, d2_mask = self._build_history(args.d2["train"], self.d2_n_users)
        self.register_buffer("d1_history", d1_hist)
        self.register_buffer("d1_history_mask", d1_mask)
        self.register_buffer("d2_history", d2_hist)
        self.register_buffer("d2_history_mask", d2_mask)

        self.loss_ce = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_history(self, train_rows, n_users):
        L = self.max_len
        history = torch.zeros(n_users, L, dtype=torch.long)
        mask = torch.zeros(n_users, L, dtype=torch.long)
        u2items = defaultdict(list)
        for row in train_rows:
            u2items[int(row[0])].append(int(row[1]) + 1)
        for u, items in u2items.items():
            kept = items[-L:]
            k = len(kept)
            if k == 0:
                continue
            history[u, :k] = torch.tensor(kept, dtype=torch.long)
            mask[u, :k] = 1
        return history, mask


    def _encode_seq(self, seq):
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        position_emb = self.position_embeddings(positions)
        mask_seq = (seq > 0).float()
        return position_emb, mask_seq

    def _domain_seq_emb(self, seq, domain):
        item_table = self.d1_item_emb if domain == 1 else self.d2_item_emb
        position_emb, mask_seq = self._encode_seq(seq)
        item_emb = item_table(seq) + position_emb
        item_shared = self.shared_layer(item_emb)
        item_specific = self.specific_layer(item_emb)
        return item_emb, item_shared, item_specific, mask_seq

    def _shared_table(self, domain):
        item_table = self.d1_item_emb if domain == 1 else self.d2_item_emb
        return self.shared_layer(item_table.weight)


    def _embedding_loss(self, d1_shared, d2_shared, d1_specific, d2_specific):
        ortho1 = torch.mean(F.cosine_similarity(d1_shared, d1_specific, dim=-1) ** 2)
        ortho2 = torch.mean(F.cosine_similarity(d2_shared, d2_specific, dim=-1) ** 2)
        ortho = ortho1 + ortho2
        mmd_shared = self._mmd_loss(d1_shared, d2_shared)
        mmd_private = 1.0 / self._mmd_loss(d1_specific, d2_specific)
        return (
            self.lambda_shared * mmd_shared
            + self.lambda_private * mmd_private
            + self.lambda_orthogonal * ortho
        )

    @staticmethod
    def _rbf_kernel(x, y, sigma=1.0):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist = x_norm - 2 * torch.matmul(x, y.T) + y_norm.T
        return torch.exp(-dist / (2 * sigma ** 2)).mean()

    def _mmd_loss(self, x, y, sigma=1.0):
        x = x.mean(dim=1)
        y = y.mean(dim=1)
        return self._rbf_kernel(x, x, sigma) + self._rbf_kernel(y, y, sigma) - 2 * self._rbf_kernel(x, y, sigma)


    def _batch_history(self, users, pos_items, domain):
        history = self.d1_history if domain == 1 else self.d2_history
        h_mask = self.d1_history_mask if domain == 1 else self.d2_history_mask
        seq = history[users].clone()
        mask = h_mask[users].clone()
        if pos_items is not None:
            pos_in_seq = (seq == (pos_items + 1).unsqueeze(1)) & (mask > 0)
            seq = torch.where(pos_in_seq, torch.zeros_like(seq), seq)
            mask = torch.where(pos_in_seq, torch.zeros_like(mask), mask)
        return seq, mask


    def _domain_diffu_loss(self, seq, target, domain):
        item_emb, item_shared, item_specific, mask_seq = self._domain_seq_emb(seq, domain)
        cond = self.LayerNorm(self.embed_dropout(item_emb))

        item_table = self.d1_item_emb if domain == 1 else self.d2_item_emb
        tag_emb = self.shared_layer(item_table(target + 1))

        rep_diffu = self.diffu(cond, tag_emb, mask_seq)
        scores = torch.matmul(rep_diffu, self._shared_table(domain).t())[:, 1:]
        loss = self.loss_ce(scores, target.long())
        return loss, item_shared, item_specific

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_seq, _ = self._batch_history(inter_d1[:, 0], inter_d1[:, 1], domain=1)
        d2_seq, _ = self._batch_history(inter_d2[:, 0], inter_d2[:, 1], domain=2)

        loss_d1, d1_shared, d1_specific = self._domain_diffu_loss(d1_seq, inter_d1[:, 1], domain=1)
        loss_d2, d2_shared, d2_specific = self._domain_diffu_loss(d2_seq, inter_d2[:, 1], domain=2)

        em_loss = self._embedding_loss(d1_shared, d2_shared, d1_specific, d2_specific)
        return 0.5 * (loss_d1 + loss_d2) + self.loss_lambda * em_loss


    @torch.no_grad()
    def _eval_domain(self, eval_set, domain, history_mask, domain_name):
        item_emb_table = self.d1_item_emb if domain == 1 else self.d2_item_emb
        shared_table = self._shared_table(domain)

        def score_fn(users):
            seq, mask = self._batch_history(users, None, domain)
            position_emb, mask_seq = self._encode_seq(seq)
            item_emb = item_emb_table(seq) + position_emb
            item_shared = self.shared_layer(item_emb)
            cond = self.LayerNorm(self.embed_dropout(item_emb))
            cond_shared = self.LayerNorm(self.embed_dropout(item_shared))
            noise_x_t = torch.randn_like(cond[:, -1, :])
            rep_diffu = self.diffu.reverse_p_sample(cond, cond_shared, noise_x_t, mask_seq)
            scores = torch.matmul(rep_diffu, shared_table.t())
            return scores[:, 1:]

        return full_rank_eval(eval_set, score_fn, history_mask, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 1, mask_d1, self.d1_domain_name)
        if domain == 2:
            return self._eval_domain(eval_set_2, 2, mask_d2, self.d2_domain_name)
        d1 = self._eval_domain(eval_set_1, 1, mask_d1, self.d1_domain_name)
        d2 = self._eval_domain(eval_set_2, 2, mask_d2, self.d2_domain_name)
        return [d1, d2]
