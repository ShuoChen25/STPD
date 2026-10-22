import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Householder import HouseHolder

from utils import full_rank_eval


class CrossAug(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params['embedding_size']
        self.shared_dim = params.get('shared_dim', 64)
        self.n_layers = params.get('n_layers', 2)
        self.reg_weight = params.get('reg_weight', 0.001)
        self.drop_rate = params.get('drop_rate', 0.3)
        self.cd_batch = params.get('cbatch_size', 512)

        self.local_aug = params.get('l_alpha', 0.7)
        self.cross_domain_aug = params.get('cd_alpha', 0.7)
        self.local_lambda = params.get('ll', 1.0)
        self.cross_domain_lambda = params.get('cdl', 1.0)
        self.align_lambda = params.get('al', 0.5)
        self.n_householder = params.get('n_hh', 8)

        self.n_shared_users = args.data['n_shared_users']
        self.d1_n_users, self.d2_n_users = args.d1['n_users'], args.d2['n_users']
        self.d1_n_items, self.d2_n_items = args.d1['n_items'], args.d2['n_items']
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.n_items = self.d1_n_items + self.d2_n_items

        self.dropout = nn.Dropout(self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        self.d1_adj = args.d1["inter_mat"]
        self.d2_adj = args.d2["inter_mat"]

        self.d1_norm_adj, self.d1_user_degree = self._get_norm_adj(self.d1_adj)
        self.d2_norm_adj, self.d2_user_degree = self._get_norm_adj(self.d2_adj)

        user_laplace = self.d1_user_degree + self.d2_user_degree + 1e-7
        self.d1_user_degree = (self.d1_user_degree / user_laplace).to(dtype=self.dtype).unsqueeze(1)
        self.d2_user_degree = (self.d2_user_degree / user_laplace).to(dtype=self.dtype).unsqueeze(1)

        if self.n_householder:
            self.d1_hh = nn.ModuleList([
                HouseHolder(args, self.shared_dim, self.n_householder)
                for _ in range(self.n_layers + 1)
            ])
            self.d2_hh = nn.ModuleList([
                HouseHolder(args, self.shared_dim, self.n_householder)
                for _ in range(self.n_layers + 1)
            ])

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)


    def _get_norm_adj(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1)).flatten()
        users_degree = torch.from_numpy(rowsum[:self.n_users]).to(self.device)
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        adj = adj_mat.dot(r_mat_inv).transpose().dot(r_mat_inv)
        adj = adj.tocoo()
        indices = torch.LongTensor(np.array([adj.row, adj.col]))
        data = torch.tensor(adj.data, dtype=self.dtype)
        norm_adj = torch.sparse_coo_tensor(
            indices, data, torch.Size(adj.shape),
            dtype=self.dtype, device=self.device,
        ).coalesce()
        return norm_adj, users_degree


    def get_emb(self):
        d1_emb = torch.cat([self.d1_user.weight, self.d1_item.weight])
        d2_emb = torch.cat([self.d2_user.weight, self.d2_item.weight])
        return d1_emb, d2_emb


    def graph_layer(self, adj_matrix, all_emb):
        side_emb = torch.sparse.mm(adj_matrix, all_emb)
        new_emb = side_emb + torch.mul(all_emb, side_emb)
        new_emb = all_emb + new_emb
        new_emb = self.dropout(new_emb)
        return new_emb


    def transfer_layer(self, d1_emb, d2_emb):
        d1_user_emb, d1_item_emb = torch.split(d1_emb, [self.n_users, self.d1_n_items])
        d2_user_emb, d2_item_emb = torch.split(d2_emb, [self.n_users, self.d2_n_items])

        common = self.d1_user_degree * d1_user_emb + self.d2_user_degree * d2_user_emb

        d1_user_emb = (common + d1_user_emb) / 2
        d2_user_emb = (common + d2_user_emb) / 2

        d1_all_emb = torch.cat([d1_user_emb, d1_item_emb], dim=0)
        d2_all_emb = torch.cat([d2_user_emb, d2_item_emb], dim=0)

        return d1_all_emb, d2_all_emb


    def graph_transfer(self, d1_emb, d2_emb):
        d1_sh, d1_sp = torch.split(d1_emb, [self.shared_dim, self.emb_dim - self.shared_dim], -1)
        d2_sh, d2_sp = torch.split(d2_emb, [self.shared_dim, self.emb_dim - self.shared_dim], -1)
        d1_graph_sh = self.graph_layer(self.d1_norm_adj, d1_sh)
        d1_graph_sp = self.graph_layer(self.d1_norm_adj, d1_sp)
        d2_graph_sh = self.graph_layer(self.d2_norm_adj, d2_sh)
        d2_graph_sp = self.graph_layer(self.d2_norm_adj, d2_sp)
        d1_trans_sh, d2_trans_sh = self.transfer_layer(d1_graph_sh, d2_graph_sh)
        d1_trans_sp, d2_trans_sp = self.transfer_layer(d1_graph_sp, d2_graph_sp)
        d1_norm_sh = F.normalize(d1_trans_sh, 2, -1)
        d1_norm_sp = F.normalize(d1_trans_sp, 2, -1)
        d2_norm_sh = F.normalize(d2_trans_sh, 2, -1)
        d2_norm_sp = F.normalize(d2_trans_sp, 2, -1)
        d1_norm_emb = torch.cat([d1_norm_sh, d1_norm_sp], -1)
        d2_norm_emb = torch.cat([d2_norm_sh, d2_norm_sp], -1)
        return d1_norm_emb, d2_norm_emb


    def extract_ui(self, emb_list, domain):
        emb = torch.cat(emb_list, -1)
        if domain == 1:
            final_user = emb[:self.d1_n_users]
        else:
            final_user = torch.cat([emb[:self.n_shared_users], emb[self.d1_n_users:self.n_users]])
        final_item = emb[self.n_users:]
        return final_user, final_item


    def get_hh_emb(self, emb, domain):
        sh_total = self.shared_dim * (self.n_layers + 1)
        sp_total = (self.emb_dim - self.shared_dim) * (self.n_layers + 1)
        joint_emb, spec_emb = torch.split(emb, [sh_total, sp_total], -1)

        if domain == 1:
            joint_hh = torch.cat([
                self.d1_hh[i](joint_emb[:, self.shared_dim * i:self.shared_dim * (i + 1)])
                for i in range(self.n_layers + 1)
            ], -1)
        else:
            joint_hh = torch.cat([
                self.d2_hh[i](joint_emb[:, self.shared_dim * i:self.shared_dim * (i + 1)])
                for i in range(self.n_layers + 1)
            ], -1)

        hh_emb = torch.cat([joint_hh, spec_emb], -1)
        return hh_emb


    def forward(self):
        d1_emb, d2_emb = self.get_emb()

        d1_emb_list = [d1_emb]
        d2_emb_list = [d2_emb]

        for _ in range(self.n_layers):
            d1_emb, d2_emb = self.graph_transfer(d1_emb, d2_emb)
            d1_emb_list.append(d1_emb)
            d2_emb_list.append(d2_emb)

        d1_user, d1_item = self.extract_ui(d1_emb_list, 1)
        d2_user, d2_item = self.extract_ui(d2_emb_list, 2)

        if self.n_householder:
            d1_user = self.get_hh_emb(d1_user, 1)
            d1_item = self.get_hh_emb(d1_item, 1)
            d2_user = self.get_hh_emb(d2_user, 2)
            d2_item = self.get_hh_emb(d2_item, 2)

        return d1_user, d1_item, d2_user, d2_item


    def get_score(self, user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)


    def get_reg_loss(self, user_ids, item_ids, domain):
        d1_emb, d2_emb = self.get_emb()
        if domain == 1:
            user_emb = d1_emb[:self.d1_n_users]
            item_emb = d1_emb[self.n_users:]
        else:
            user_emb = torch.cat([d2_emb[:self.n_shared_users], d2_emb[self.d1_n_users:self.n_users]])
            item_emb = d2_emb[self.n_users:]

        user_reg_loss = (torch.norm(user_emb[user_ids]) ** 2 + torch.norm(user_emb[user_ids]) ** 2) / len(user_ids)
        item_reg_loss = (torch.norm(item_emb[item_ids]) ** 2 + torch.norm(item_emb[item_ids]) ** 2) / len(item_ids)

        return user_reg_loss + item_reg_loss


    def get_cross_items(self, item_emb, item_pos_ids, item_neg_ids):
        sh_total = self.shared_dim * (self.n_layers + 1)
        sp_total = (self.emb_dim - self.shared_dim) * (self.n_layers + 1)
        item_sh, item_sp = torch.split(item_emb, [sh_total, sp_total], -1)
        c1 = torch.cat([item_sh[item_pos_ids], item_sp[item_neg_ids]], -1)
        c2 = torch.cat([item_sh[item_neg_ids], item_sp[item_pos_ids]], -1)
        return c1, c2


    def get_cd_loss(self, ori_users, ori_items, tar_users, tar_items):
        pos_scores = self.get_score(ori_users, ori_items)
        neg_scores = self.get_score(tar_users, tar_items)

        sh_total = self.shared_dim * (self.n_layers + 1)
        sp_total = (self.emb_dim - self.shared_dim) * (self.n_layers + 1)
        ori_user_sh, ori_user_sp = torch.split(ori_users, [sh_total, sp_total], -1)
        ori_item_sh, ori_item_sp = torch.split(ori_items, [sh_total, sp_total], -1)
        tar_user_sh, tar_user_sp = torch.split(tar_users, [sh_total, sp_total], -1)
        tar_item_sh, tar_item_sp = torch.split(tar_items, [sh_total, sp_total], -1)

        c1_scores = self.get_score(ori_user_sh, ori_item_sh) + self.get_score(tar_user_sp, tar_item_sp)
        c2_scores = self.get_score(tar_user_sh, tar_item_sh) + self.get_score(ori_user_sp, ori_item_sp)

        c1_final = self.cross_domain_aug * (neg_scores - c1_scores) + (1 - self.cross_domain_aug) * (c1_scores - pos_scores)
        c2_final = self.cross_domain_aug * (neg_scores - c2_scores) + (1 - self.cross_domain_aug) * (c2_scores - pos_scores)

        labels = torch.zeros(len(ori_users), dtype=self.dtype, device=self.device)
        cd_loss = self.loss(self.sigmoid(c1_final), labels) + self.loss(self.sigmoid(c2_final), labels)
        return cd_loss


    def get_align_loss(self, d1_emb, d2_emb):
        sh_total = self.shared_dim * (self.n_layers + 1)
        sp_total = (self.emb_dim - self.shared_dim) * (self.n_layers + 1)
        d1_joint_emb, _ = torch.split(d1_emb, [sh_total, sp_total], -1)
        d2_joint_emb, _ = torch.split(d2_emb, [sh_total, sp_total], -1)

        mu_diff = torch.mean(d1_joint_emb, 0) - torch.mean(d2_joint_emb, 0)
        return torch.norm(mu_diff) ** 2


    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()

        # Domain 1 Positive inters
        d1_users, d1_items = d1_user_emb[inter_d1[:, 0]], d1_item_emb[inter_d1[:, 1]]
        d1_pos_label = torch.ones(len(inter_d1), dtype=self.dtype, device=self.device)
        d1_output = self.sigmoid(self.get_score(d1_users, d1_items))
        d1_loss = self.loss(d1_output, d1_pos_label)
        d1_loss = d1_loss + self.reg_weight * self.get_reg_loss(inter_d1[:, 0], inter_d1[:, 1], 1)

        # Domain 2 Positive inters
        d2_users, d2_items = d2_user_emb[inter_d2[:, 0]], d2_item_emb[inter_d2[:, 1]]
        d2_pos_label = torch.ones(len(inter_d2), dtype=self.dtype, device=self.device)
        d2_output = self.sigmoid(self.get_score(d2_users, d2_items))
        d2_loss = self.loss(d2_output, d2_pos_label)
        d2_loss = d2_loss + self.reg_weight * self.get_reg_loss(inter_d2[:, 0], inter_d2[:, 1], 2)

        # Align loss
        align_loss = 0
        if self.align_lambda:
            align_loss += self.align_lambda * self.get_align_loss(
                d1_user_emb[:self.n_shared_users], d2_user_emb[:self.n_shared_users])
            align_loss += self.align_lambda * self.get_align_loss(d1_items, d2_items)

        # Negative inters (single neg per sample, adapted to Copenhagen pipeline)
        d1_neg_label = torch.zeros(len(inter_d1), dtype=self.dtype, device=self.device)
        d2_neg_label = torch.zeros(len(inter_d2), dtype=self.dtype, device=self.device)

        d1_neg_items = d1_item_emb[neg_d1_item]
        d1_neg_output = self.sigmoid(self.get_score(d1_users, d1_neg_items))
        d1_loss = d1_loss + self.loss(d1_neg_output, d1_neg_label)
        d1_loss = d1_loss + self.reg_weight * self.get_reg_loss(inter_d1[:, 0], neg_d1_item, 1)

        d2_neg_items = d2_item_emb[neg_d2_item]
        d2_neg_output = self.sigmoid(self.get_score(d2_users, d2_neg_items))
        d2_loss = d2_loss + self.loss(d2_neg_output, d2_neg_label)
        d2_loss = d2_loss + self.reg_weight * self.get_reg_loss(inter_d2[:, 0], neg_d2_item, 2)

        # Cross local integrated samples
        if self.local_lambda:
            d1_c1, d1_c2 = self.get_cross_items(d1_item_emb, inter_d1[:, 1], neg_d1_item)
            d1_c1_items = self.local_aug * (d1_neg_items - d1_c1) + (1 - self.local_aug) * (d1_c1 - d1_items)
            d1_c1_output = self.sigmoid(self.get_score(d1_users, d1_c1_items))
            d1_loss = d1_loss + self.loss(d1_c1_output, d1_neg_label)
            d1_c2_items = self.local_aug * (d1_neg_items - d1_c2) + (1 - self.local_aug) * (d1_c2 - d1_items)
            d1_c2_output = self.sigmoid(self.get_score(d1_users, d1_c2_items))
            d1_loss = d1_loss + self.local_lambda * self.loss(d1_c2_output, d1_neg_label)

            d2_c1, d2_c2 = self.get_cross_items(d2_item_emb, inter_d2[:, 1], neg_d2_item)
            d2_c1_items = self.local_aug * (d2_neg_items - d2_c1) + (1 - self.local_aug) * (d2_c1 - d2_items)
            d2_c1_output = self.sigmoid(self.get_score(d2_users, d2_c1_items))
            d2_loss = d2_loss + self.loss(d2_c1_output, d2_neg_label)
            d2_c2_items = self.local_aug * (d2_neg_items - d2_c2) + (1 - self.local_aug) * (d2_c2 - d2_items)
            d2_c2_output = self.sigmoid(self.get_score(d2_users, d2_c2_items))
            d2_loss = d2_loss + self.local_lambda * self.loss(d2_c2_output, d2_neg_label)

        # Cross domain integrated samples
        if self.cross_domain_lambda:
            d1_cd_ids = np.random.randint(len(inter_d1), size=self.cd_batch)
            d2_cd_ids = np.random.randint(len(inter_d2), size=self.cd_batch)

            d1_cd_users = d1_users[d1_cd_ids]
            d1_cd_items = d1_items[d1_cd_ids]
            d1_cd_neg_items = d1_neg_items[d1_cd_ids]

            d2_cd_users = d2_users[d2_cd_ids]
            d2_cd_items = d2_items[d2_cd_ids]
            d2_cd_neg_items = d2_neg_items[d2_cd_ids]

            d1_cd_loss = self.get_cd_loss(d1_cd_users, d1_cd_items, d2_cd_users, d2_cd_neg_items)
            d1_loss = d1_loss + self.cross_domain_lambda * d1_cd_loss

            d2_cd_loss = self.get_cd_loss(d2_cd_users, d2_cd_items, d1_cd_users, d1_cd_neg_items)
            d2_loss = d2_loss + self.cross_domain_lambda * d2_cd_loss

        total_loss = (d1_loss + d2_loss) / 2 + align_loss

        return total_loss


    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        """Full in-domain ranking with user-history masking."""
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)


    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user_emb, d1_item_emb, self.d1_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user_emb, d2_item_emb, self.d2_domain_name, mode, mask_d2)
        d1_metrics = self._eval_domain(eval_set_1, d1_user_emb, d1_item_emb, self.d1_domain_name, mode, mask_d1)
        d2_metrics = self._eval_domain(eval_set_2, d2_user_emb, d2_item_emb, self.d2_domain_name, mode, mask_d2)
        return [d1_metrics, d2_metrics]
