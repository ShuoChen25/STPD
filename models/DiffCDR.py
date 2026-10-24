import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import full_rank_eval
from .DiffCDR_dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


_noise_schedule = NoiseScheduleVP(schedule="linear")


def get_timestep_embedding(timesteps, embedding_dim: int):
    timesteps = timesteps.to(dtype=torch.float32)
    assert len(timesteps.shape) == 1
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb


class _DiffusionModel(nn.Module):

    def __init__(
        self,
        num_steps=200,
        diff_dim=32,
        input_dim=32,
        c_scale=0.1,
        diff_sample_steps=30,
        diff_task_lambda=0.1,
        diff_mask_rate=0.1,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert (
            self.alphas.shape
            == self.alphas_prod.shape
            == self.alphas_prod_p.shape
            == self.alphas_bar_sqrt.shape
            == self.one_minus_alphas_bar_log.shape
            == self.one_minus_alphas_bar_sqrt.shape
        )

        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, diff_dim),
                nn.Linear(diff_dim, diff_dim),
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.step_emb_linear = nn.ModuleList(
            [
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [
                nn.Linear(input_dim, input_dim),
            ]
        )

        self.num_layers = 1
        self.al_linear = nn.Linear(input_dim, input_dim, False)

    def forward(self, x, t, cond_emb, cond_mask):
        for idx in range(self.num_layers):
            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)

            cond_embedding = self.cond_emb_linear[idx](cond_emb)

            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb

            x = self.linears[0](x)
            x = self.linears[1](x)
            x = self.linears[2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


def _q_x_fn(model, x_0, t, device):
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise


def _diffusion_noise_loss_fn(model, x_0, cond_emb, device):
    num_steps = model.num_steps
    mask_rate = model.mask_rate

    batch_size = x_0.shape[0]
    t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
    if batch_size % 2 == 0:
        t = torch.cat([t, num_steps - 1 - t], dim=0)
    else:
        extra_t = torch.randint(0, num_steps, size=(1,), device=device)
        t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
    t = t.unsqueeze(-1)

    x, e = _q_x_fn(model, x_0, t, device)

    cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
    cond_mask = 1 - cond_mask.int()

    output = model(x, t.squeeze(-1), cond_emb, cond_mask)
    return F.smooth_l1_loss(e, output)


def _p_sample(model, cond_emb, x, iid_emb, device):
    model_kwargs = {
        "cond_emb": cond_emb,
        "cond_mask": torch.zeros(cond_emb.size()[0], device=device),
    }

    model_fn = model_wrapper(
        model,
        _noise_schedule,
        is_cond_classifier=True,
        classifier_scale=model.c_scale,
        time_input_type="1",
        total_N=model.num_steps,
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, _noise_schedule)

    sample = dpm_solver.sample(
        x,
        steps=model.sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )

    return model.get_al_emb(sample).to(device), iid_emb


def _p_sample_loop(model, cond_emb, iid_input, device):
    cur_x = cond_emb
    cur_x, iid_emb_out = _p_sample(model, cond_emb, cur_x, iid_input, device)
    return cur_x, iid_emb_out


class _MFDomain(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, dtype):
        super().__init__()
        self.uid_embedding = nn.Embedding(n_users, emb_dim, dtype=dtype)
        self.iid_embedding = nn.Embedding(n_items, emb_dim, dtype=dtype)

    def score(self, user, item):
        uid_emb = self.uid_embedding(user)
        iid_emb = self.iid_embedding(item)
        return torch.sum(uid_emb * iid_emb, dim=1)


class DiffCDR(nn.Module):

    disable_compile = True

    def __init__(self, args):
        super().__init__()
        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = int(params.get("diffcdr_emb_dim", params.get("emb_dim", 10)))
        self.pretrain_epochs = int(params.get("diffcdr_pretrain_epochs", 50))
        self.reg_weight = float(params.get("diffcdr_reg_weight", params.get("reg_weight", 0.0)))
        self.diff_clip_norm = float(params.get("diffcdr_clip_norm", 1.0))

        self.diff_steps = int(params.get("diffcdr_steps", params.get("diff_steps", 1000)))
        self.diff_scale = float(params.get("diffcdr_scale", params.get("diff_scale", 0.1)))
        self.diff_dim = int(params.get("diffcdr_dim", params.get("diff_dim", 128)))
        self.diff_sample_steps = int(params.get("diffcdr_sample_steps", params.get("diff_sample_steps", 30)))
        self.diff_task_lambda = float(params.get("diffcdr_task_lambda", params.get("diff_task_lambda", 0.1)))
        self.diff_mask_rate = float(params.get("diffcdr_mask_rate", params.get("diff_mask_rate", 0.1)))

        self.n_shared_users = int(args.data["n_shared_users"])
        self.d1_n_users = int(args.d1["n_users"])
        self.d2_n_users = int(args.d2["n_users"])
        self.d1_n_items = int(args.d1["n_items"])
        self.d2_n_items = int(args.d2["n_items"])
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.d1_model = _MFDomain(self.n_users, self.d1_n_items, self.emb_dim, self.dtype)
        self.d2_model = _MFDomain(self.n_users, self.d2_n_items, self.emb_dim, self.dtype)
        self.d2_to_d1 = _DiffusionModel(
            self.diff_steps,
            self.diff_dim,
            self.emb_dim,
            self.diff_scale,
            self.diff_sample_steps,
            self.diff_task_lambda,
            self.diff_mask_rate,
        )
        self.d1_to_d2 = _DiffusionModel(
            self.diff_steps,
            self.diff_dim,
            self.emb_dim,
            self.diff_scale,
            self.diff_sample_steps,
            self.diff_task_lambda,
            self.diff_mask_rate,
        )

        self._epoch = [0]
        self.min_train_epochs = self.pretrain_epochs + 3
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def configure_optimizers(self, params, default_lr):
        wd = float(params.get("diffcdr_wd", params.get("weight_decay", 0.0)) or 0.0)
        base_lr = float(params.get("diffcdr_base_lr", default_lr))
        diff_lr = float(params.get("diffcdr_diff_lr", params.get("diff_lr", 0.001)))
        return {
            "base": torch.optim.Adam(
                list(self.d1_model.parameters()) + list(self.d2_model.parameters()),
                lr=base_lr,
                weight_decay=wd,
            ),
            "diff": torch.optim.Adam(
                list(self.d2_to_d1.parameters()) + list(self.d1_to_d2.parameters()),
                lr=diff_lr,
            ),
        }

    def _to_union(self, users, domain):
        if domain == 1:
            return users.long()
        return torch.where(
            users < self.n_shared_users,
            users,
            users + (self.d1_n_users - self.n_shared_users),
        ).long()

    @staticmethod
    def _bpr(pos_score, neg_score):
        return -F.logsigmoid(pos_score - neg_score).mean()

    def _base_domain_loss(self, inter, neg_item, domain):
        model = self.d1_model if domain == 1 else self.d2_model
        users = self._to_union(inter[:, 0], domain)
        pos_item = inter[:, 1].long()
        neg_item = neg_item.long()

        pos_score = model.score(users, pos_item)
        neg_score = model.score(users, neg_item)
        loss = self._bpr(pos_score, neg_score)
        if self.reg_weight > 0:
            u = model.uid_embedding(users)
            pi = model.iid_embedding(pos_item)
            ni = model.iid_embedding(neg_item)
            loss = loss + self.reg_weight * (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / (2 * len(users))
        return loss

    def _base_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        return 0.5 * (
            self._base_domain_loss(inter_d1, neg_d1_item, 1)
            + self._base_domain_loss(inter_d2, neg_d2_item, 2)
        )

    def _overlap_batch(self, inter, neg_item, domain):
        users = self._to_union(inter[:, 0], domain)
        keep = users < self.n_shared_users
        return users[keep], inter[keep, 1].long(), neg_item[keep].long()

    def _diff_direction_loss(self, users, pos_item, neg_item, target_domain, is_task):
        if users.numel() == 0:
            return None

        if target_domain == 1:
            target_model, source_model = self.d1_model, self.d2_model
            diff_model = self.d2_to_d1
        else:
            target_model, source_model = self.d2_model, self.d1_model
            diff_model = self.d1_to_d2

        x_0 = target_model.uid_embedding(users).detach()
        cond_emb = source_model.uid_embedding(users).detach()

        if not is_task:
            return _diffusion_noise_loss_fn(diff_model, x_0, cond_emb, self.device)

        pos_iid_emb = target_model.iid_embedding(pos_item).detach()
        neg_iid_emb = target_model.iid_embedding(neg_item).detach()
        final_output, pos_iid_emb = _p_sample_loop(diff_model, cond_emb, pos_iid_emb, self.device)
        pos_score = torch.sum(final_output * pos_iid_emb, dim=1)
        neg_score = torch.sum(final_output * neg_iid_emb, dim=1)
        task_loss = self._bpr(pos_score, neg_score)
        return F.smooth_l1_loss(x_0, final_output) + diff_model.task_lambda * task_loss

    def _diff_losses(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item, is_task):
        d1_users, d1_pos, d1_neg = self._overlap_batch(inter_d1, neg_d1_item, 1)
        d2_users, d2_pos, d2_neg = self._overlap_batch(inter_d2, neg_d2_item, 2)

        losses = []
        d1_loss = self._diff_direction_loss(d1_users, d1_pos, d1_neg, target_domain=1, is_task=is_task)
        d2_loss = self._diff_direction_loss(d2_users, d2_pos, d2_neg, target_domain=2, is_task=is_task)
        if d1_loss is not None:
            losses.append(d1_loss)
        if d2_loss is not None:
            losses.append(d2_loss)

        if not losses:
            return None
        return torch.stack(losses).mean()

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        if self._epoch[0] < self.pretrain_epochs:
            return self._base_loss(inter_d1, neg_d1_item, inter_d2, neg_d2_item)
        noise_loss = self._diff_losses(inter_d1, neg_d1_item, inter_d2, neg_d2_item, is_task=False)
        task_loss = self._diff_losses(inter_d1, neg_d1_item, inter_d2, neg_d2_item, is_task=True)
        zero = self.d1_model.uid_embedding.weight.new_zeros(())
        return (noise_loss if noise_loss is not None else zero) + (task_loss if task_loss is not None else zero)

    def train_batch(self, optimizers, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        if self._epoch[0] < self.pretrain_epochs:
            optimizers["base"].zero_grad()
            loss = self._base_loss(inter_d1, neg_d1_item, inter_d2, neg_d2_item)
            loss.backward()
            optimizers["base"].step()
            return loss.detach()

        zero = self.d1_model.uid_embedding.weight.new_zeros(())
        noise_loss = self._diff_losses(inter_d1, neg_d1_item, inter_d2, neg_d2_item, is_task=False)
        if noise_loss is not None:
            optimizers["diff"].zero_grad()
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.d2_to_d1.parameters()) + list(self.d1_to_d2.parameters()),
                self.diff_clip_norm,
            )
            optimizers["diff"].step()

        task_loss = self._diff_losses(inter_d1, neg_d1_item, inter_d2, neg_d2_item, is_task=True)
        if task_loss is not None:
            optimizers["diff"].zero_grad()
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.d2_to_d1.parameters()) + list(self.d1_to_d2.parameters()),
                self.diff_clip_norm,
            )
            optimizers["diff"].step()

        out = (noise_loss.detach() if noise_loss is not None else zero) + (
            task_loss.detach() if task_loss is not None else zero
        )
        return out

    @torch.no_grad()
    def _user_for_eval(self, users, domain):
        union = self._to_union(users, domain)
        target_model = self.d1_model if domain == 1 else self.d2_model
        source_model = self.d2_model if domain == 1 else self.d1_model
        diff_model = self.d2_to_d1 if domain == 1 else self.d1_to_d2

        raw = target_model.uid_embedding(union)
        if self._epoch[0] < self.pretrain_epochs or self.n_shared_users == 0:
            return raw

        overlap = union < self.n_shared_users
        if not torch.any(overlap):
            return raw

        out = raw.clone()
        cond = source_model.uid_embedding(union[overlap])
        dummy_iid = target_model.iid_embedding.weight[:1].expand(cond.size(0), -1)
        trans_emb, _ = _p_sample_loop(diff_model, cond, dummy_iid, self.device)
        out[overlap] = trans_emb
        return out

    @torch.no_grad()
    def _eval_domain(self, eval_set, domain, history, domain_name):
        item_emb = self.d1_model.iid_embedding.weight if domain == 1 else self.d2_model.iid_embedding.weight

        def score_fn(users):
            user_emb = self._user_for_eval(users, domain)
            return user_emb @ item_emb.t()

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
