import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval, symmetric_normalize


class LightGCN(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = 64
        self.n_layers = 3
        self.drop_rate = params["drop_rate"]
        self.reg_weight = params.get("reg_weight", 0.001)

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

        self.d1_graph = self._build_norm_graph(args.d1["inter_mat"])
        self.d2_graph = self._build_norm_graph(args.d2["inter_mat"])

        self.neg_ratio = params.get("neg_ratio", 1)

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def _build_norm_graph(self, sp_mat):
        norm = symmetric_normalize(sp_mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[: self.d1_n_users], emb[self.n_users:]
        user = torch.cat(
            [emb[: self.n_shared_users], emb[self.d1_n_users: self.n_users]], dim=0
        )
        return user, emb[self.n_users:]


    def aggregate(self, graph, all_emb):
        embeddings = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embeddings.append(all_emb)
        return torch.stack(embeddings, dim=1).mean(dim=1)


    def forward(self):
        d1_emb = torch.cat([self.d1_user.weight, self.d1_item.weight])
        d2_emb = torch.cat([self.d2_user.weight, self.d2_item.weight])

        d1_emb = self.aggregate(self.d1_graph, d1_emb)
        d2_emb = self.aggregate(self.d2_graph, d2_emb)

        d1_user, d1_item = self._split_domain_ui(d1_emb, 1)
        d2_user, d2_item = self._split_domain_ui(d2_emb, 2)

        return d1_user, d1_item, d2_user, d2_item


    def get_score(self, user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)

    def get_bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos = self.get_score(user_emb, pos_item_emb)
        neg = self.get_score(user_emb, neg_item_emb)
        return -F.logsigmoid(pos - neg).mean()

    def get_bpr_reg_loss(self, users, pos_items, neg_items, domain):
        if domain == 1:
            u = self.d1_user.weight[users]
            pi = self.d1_item.weight[pos_items]
            ni = self.d1_item.weight[neg_items]
        else:
            u = self.d2_user.weight[users]
            pi = self.d2_item.weight[pos_items]
            ni = self.d2_item.weight[neg_items]
        return (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / (
            2 * len(users)
        )


    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_user, d1_item, d2_user, d2_item = self.forward()

        d1_u = d1_user[inter_d1[:, 0]]
        d1_pos = d1_item[inter_d1[:, 1]]
        d2_u = d2_user[inter_d2[:, 0]]
        d2_pos = d2_item[inter_d2[:, 1]]

        d1_neg = d1_item[neg_d1_item]
        d2_neg = d2_item[neg_d2_item]

        d1_loss = self.get_bpr_loss(d1_u, d1_pos, d1_neg)
        d1_loss += self.reg_weight * self.get_bpr_reg_loss(
            inter_d1[:, 0], inter_d1[:, 1], neg_d1_item, 1
        )

        d2_loss = self.get_bpr_loss(d2_u, d2_pos, d2_neg)
        d2_loss += self.reg_weight * self.get_bpr_reg_loss(
            inter_d2[:, 0], inter_d2[:, 1], neg_d2_item, 2
        )

        return (d1_loss + d2_loss) / 2


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
