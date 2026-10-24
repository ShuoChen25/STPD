import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval, symmetric_normalize


def _uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


class _CDUserItemAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.W_q_user = nn.Linear(embed_dim, embed_dim)
        self.W_k_user = nn.Linear(embed_dim, embed_dim)
        self.W_v_user = nn.Linear(embed_dim, embed_dim)
        self.W_q_item = nn.Linear(embed_dim, embed_dim)
        self.W_k_item = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, Q_user, K_user, V_user, Q_item, K_item, alpha=0.9):
        q_u = self.W_q_user(Q_user)
        k_u = self.W_k_user(K_user)
        v_u = self.W_v_user(V_user)
        attn_u = F.softmax(q_u @ k_u.transpose(0, 1) / self.scale, dim=1)

        q_i = self.W_q_item(Q_item)
        k_i = self.W_k_item(K_item)
        attn_i = F.softmax(q_i @ k_i.transpose(0, 1) / self.scale, dim=1)

        attn = alpha * attn_u + (1 - alpha) * attn_i
        return attn @ v_u


class VUG(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params.get("embedding_size", 64)
        self.n_layers = int(params.get("vug_n_layers", params.get("n_layers", 2)))
        self.reg_weight = float(params.get("vug_reg_weight", params.get("reg_weight", 1e-4)))
        self.drop_rate = float(params.get("vug_drop_rate", params.get("drop_rate", 0.0)))
        self.lambda_source = float(params.get("vug_lambda_source", 0.5))
        self.lambda_target = float(params.get("vug_lambda_target", 0.5))
        self.connect_way = params.get("vug_connect_way", "mean")
        self.is_transfer = bool(params.get("vug_is_transfer", True))

        self.gen_weight = float(params.get("vug_gen_weight", 0.9))
        self.enhance_mode = params.get("vug_enhance_mode", "add")
        self.enhance_weight = float(params.get("vug_enhance_weight", 0.5))
        self.user_weight_attn = float(params.get("vug_user_weight_attn", 0.9))
        self.gen_loss_weight = float(params.get("vug_gen_loss_weight", 1.0))
        self.gen_chunk_size = int(params.get("vug_gen_chunk_size", 8192))

        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users = args.d1["n_users"]
        self.d2_n_users = args.d2["n_users"]
        self.d1_n_items = args.d1["n_items"]
        self.d2_n_items = args.d2["n_items"]
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        self.dropout = nn.Dropout(self.drop_rate)
        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.gen_mse = nn.MSELoss()

        self.d1_graph = self._build_norm_graph(args.d1["inter_mat"])
        self.d2_graph = self._build_norm_graph(args.d2["inter_mat"])

        d1_u_deg = self._user_degree(args.d1["inter_mat"])
        d2_u_deg = self._user_degree(args.d2["inter_mat"])
        self.register_buffer("d1_user_deg", d1_u_deg)
        self.register_buffer("d2_user_deg", d2_u_deg)

        mask = torch.zeros(self.n_users, 1, dtype=self.dtype)
        mask[self.d1_n_users:self.n_users] = 1.0
        self.register_buffer("target_only_mask", mask)

        self.source_generator = _CDUserItemAttention(self.emb_dim)

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_norm_graph(self, sp_mat):
        norm = symmetric_normalize(sp_mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _user_degree(self, sp_mat):
        deg = np.array(sp_mat.sum(1)).flatten()[: self.n_users]
        return torch.tensor(deg, dtype=self.dtype).unsqueeze(1)


    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[: self.d1_n_users], emb[self.n_users:]
        user = torch.cat(
            [emb[: self.n_shared_users], emb[self.d1_n_users: self.n_users]], dim=0
        )
        return user, emb[self.n_users:]

    def _d2_to_union(self, d2_user_ids):
        return torch.where(
            d2_user_ids < self.n_shared_users,
            d2_user_ids,
            d2_user_ids + (self.d1_n_users - self.n_shared_users),
        )


    def _item_level_user_emb(self, user_ids):
        ego = torch.cat([self.d2_user.weight, self.d2_item.weight], dim=0)
        side = torch.sparse.mm(self.d2_graph, ego)
        user_agg, _ = torch.split(side, [self.n_users, self.d2_n_items])
        return user_agg[user_ids]

    def generate_source_embeddings(self, user_ids=None):
        if user_ids is None:
            user_ids = torch.arange(self.n_users, device=self.d2_user.weight.device)

        Q_user = self.d2_user(user_ids)
        Q_item = self._item_level_user_emb(user_ids)

        overlap_idx = torch.arange(self.n_shared_users, device=user_ids.device)
        if overlap_idx.numel() == 0:
            return torch.zeros_like(Q_user)

        K_user = self.d2_user(overlap_idx)
        K_item = self._item_level_user_emb(overlap_idx)
        V_user = self.d1_user(overlap_idx)

        n = Q_user.shape[0]
        chunk = self.gen_chunk_size
        if chunk <= 0 or chunk >= n:
            return self.source_generator(Q_user, K_user, V_user, Q_item, K_item, alpha=self.user_weight_attn)

        outs = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            outs.append(
                self.source_generator(
                    Q_user[start:end], K_user, V_user, Q_item[start:end], K_item,
                    alpha=self.user_weight_attn,
                )
            )
        return torch.cat(outs, dim=0)


    def get_ego_embeddings(self, domain):
        if domain == 1:
            user_emb = self.d1_user.weight
            item_emb = self.d1_item.weight
            adj = self.d1_graph
            if self.enhance_mode == "asrealsource":
                gen_src = self.generate_source_embeddings()
                user_emb = user_emb + self.enhance_weight * self.target_only_mask * gen_src
        else:
            user_emb = self.d2_user.weight
            item_emb = self.d2_item.weight
            adj = self.d2_graph
            if self.enhance_mode == "add":
                gen_src = self.generate_source_embeddings()
                user_emb = user_emb + self.enhance_weight * self.target_only_mask * gen_src
        return torch.cat([user_emb, item_emb], dim=0), adj

    def graph_layer(self, adj, all_emb):
        side = torch.sparse.mm(adj, all_emb)
        out = side + torch.mul(all_emb, side)
        out = all_emb + out
        return self.dropout(out)

    def transfer_layer(self, d1_emb, d2_emb):
        d1_u, d1_i = torch.split(d1_emb, [self.n_users, self.d1_n_items])
        d2_u, d2_i = torch.split(d2_emb, [self.n_users, self.d2_n_items])

        d1_u_lam = self.lambda_source * d1_u + (1 - self.lambda_source) * d2_u
        d2_u_lam = self.lambda_target * d2_u + (1 - self.lambda_target) * d1_u

        u_lap = self.d1_user_deg + self.d2_user_deg + 1e-7
        u_avg = (self.d1_user_deg * d1_u + self.d2_user_deg * d2_u) / u_lap

        d1_u_avg = (d1_u_lam + u_avg) / 2
        d2_u_avg = (d2_u_lam + u_avg) / 2

        n_sh = self.n_shared_users
        d1_n = self.d1_n_users
        if self.enhance_mode == "asrealsource":
            new_d1_u = torch.cat([d1_u_avg[:n_sh], d1_u[n_sh:d1_n], d1_u_avg[d1_n:]], dim=0)
            new_d2_u = torch.cat([d2_u_avg[:n_sh], d2_u[n_sh:d1_n], d2_u_avg[d1_n:]], dim=0)
        else:
            new_d1_u = torch.cat([d1_u_avg[:n_sh], d1_u[n_sh:]], dim=0)
            new_d2_u = torch.cat([d2_u_avg[:n_sh], d2_u[n_sh:]], dim=0)

        return torch.cat([new_d1_u, d1_i], dim=0), torch.cat([new_d2_u, d2_i], dim=0)

    def forward(self):
        d1_all, d1_adj = self.get_ego_embeddings(1)
        d2_all, d2_adj = self.get_ego_embeddings(2)

        d1_layers = [d1_all]
        d2_layers = [d2_all]
        for _ in range(self.n_layers):
            d1_all = self.graph_layer(d1_adj, d1_all)
            d2_all = self.graph_layer(d2_adj, d2_all)
            if self.is_transfer:
                d1_all, d2_all = self.transfer_layer(d1_all, d2_all)
            d1_layers.append(F.normalize(d1_all, p=2, dim=1))
            d2_layers.append(F.normalize(d2_all, p=2, dim=1))

        if self.connect_way == "concat":
            d1_emb = torch.cat(d1_layers, dim=1)
            d2_emb = torch.cat(d2_layers, dim=1)
        else:
            d1_emb = torch.stack(d1_layers, dim=1).mean(dim=1)
            d2_emb = torch.stack(d2_layers, dim=1).mean(dim=1)

        d1_user, d1_item = self._split_domain_ui(d1_emb, 1)
        d2_user, d2_item = self._split_domain_ui(d2_emb, 2)
        return d1_user, d1_item, d2_user, d2_item


    @staticmethod
    def get_score(user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)

    @staticmethod
    def _reg_loss(*embs):
        loss = sum((e ** 2).sum() for e in embs) / 2
        return loss / embs[0].shape[0]

    def calculate_gen_loss(self, union_user_ids):
        gen = self.generate_source_embeddings(union_user_ids)
        overlap_mask = union_user_ids < self.n_shared_users
        if overlap_mask.sum() == 0:
            return torch.tensor(0.0, device=union_user_ids.device)
        gen_ov = gen[overlap_mask]
        real_ov = self.d1_user(union_user_ids[overlap_mask])
        return self.gen_weight * self.gen_mse(gen_ov, real_ov) + (1 - self.gen_weight) * _uniformity(gen_ov)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_user, d1_item, d2_user, d2_item = self.forward()

        d1_u_idx, d1_i_idx = inter_d1[:, 0], inter_d1[:, 1]
        d1_u = d1_user[d1_u_idx]
        d1_pos = d1_item[d1_i_idx]
        d1_neg = d1_item[neg_d1_item]
        d1_pos_s = self.sigmoid(self.get_score(d1_u, d1_pos))
        d1_neg_s = self.sigmoid(self.get_score(d1_u, d1_neg))
        d1_loss = self.bce(d1_pos_s, torch.ones_like(d1_pos_s)) + self.bce(d1_neg_s, torch.zeros_like(d1_neg_s))
        d1_loss = d1_loss + self.reg_weight * self._reg_loss(
            self.d1_user.weight[d1_u_idx],
            self.d1_item.weight[d1_i_idx],
            self.d1_item.weight[neg_d1_item],
        )

        d2_u_idx, d2_i_idx = inter_d2[:, 0], inter_d2[:, 1]
        d2_u = d2_user[d2_u_idx]
        d2_pos = d2_item[d2_i_idx]
        d2_neg = d2_item[neg_d2_item]
        d2_pos_s = self.sigmoid(self.get_score(d2_u, d2_pos))
        d2_neg_s = self.sigmoid(self.get_score(d2_u, d2_neg))
        d2_loss = self.bce(d2_pos_s, torch.ones_like(d2_pos_s)) + self.bce(d2_neg_s, torch.zeros_like(d2_neg_s))
        d2_u_union = self._d2_to_union(d2_u_idx)
        d2_loss = d2_loss + self.reg_weight * self._reg_loss(
            self.d2_user.weight[d2_u_union],
            self.d2_item.weight[d2_i_idx],
            self.d2_item.weight[neg_d2_item],
        )

        gen_loss = self.calculate_gen_loss(d2_u_union)

        return (d1_loss + d2_loss) / 2 + self.gen_loss_weight * gen_loss


    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user, d1_item, d2_user, d2_item = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        d1 = self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        d2 = self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        return [d1, d2]
