import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval


class BiTGCF(nn.Module):
    """BiTGCF adapted from VUG00/recbole_cdr/model/cross_domain_recommender/bitgcf.py.

    Source domain = d1, target domain = d2. The model keeps BiTGCF's original
    pointwise BCE objective, graph propagation, and bidirectional transfer
    logic, while consuming Copenhagen's shared train/valid/test pipeline.
    """

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = int(params.get("bitgcf_embedding_size", 64))
        self.n_layers = int(params.get("bitgcf_n_layers", 2))
        self.reg_weight = float(params.get("bitgcf_reg_weight", 1e-3))
        self.lambda_source = float(params.get("bitgcf_lambda_source", 0.8))
        self.lambda_target = float(params.get("bitgcf_lambda_target", 0.8))
        self.drop_rate = float(params.get("bitgcf_drop_rate", 0.3))
        self.connect_way = params.get("bitgcf_connect_way", "concat")
        self.is_transfer = bool(params.get("bitgcf_is_transfer", True))

        self.n_shared_users = int(args.data["n_shared_users"])
        self.d1_n_users = int(args.d1["n_users"])
        self.d2_n_users = int(args.d2["n_users"])
        self.d1_n_items = int(args.d1["n_items"])
        self.d2_n_items = int(args.d2["n_items"])
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users
        self.n_items = self.d1_n_items + self.d2_n_items
        self.n_shared_items = int(args.data.get("n_shared_items", 0) or 0)
        self.d1_item_offset = self.d2_n_items - self.n_shared_items

        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        self.d1_user_embedding = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_user_embedding = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_embedding = nn.Embedding(self.n_items, self.emb_dim, dtype=self.dtype)
        self.d2_item_embedding = nn.Embedding(self.n_items, self.emb_dim, dtype=self.dtype)

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        d1_inter_mat = self._build_inter_matrix(args.d1["train"], domain=1)
        d2_inter_mat = self._build_inter_matrix(args.d2["train"], domain=2)
        self.d1_graph = self._build_norm_adj(d1_inter_mat)
        self.d2_graph = self._build_norm_adj(d2_inter_mat)

        self.register_buffer("d1_user_degree", self._degree_tensor(d1_inter_mat.sum(axis=1)))
        self.register_buffer("d2_user_degree", self._degree_tensor(d2_inter_mat.sum(axis=1)))
        self.register_buffer("d1_item_degree", self._degree_tensor(d1_inter_mat.sum(axis=0)).t())
        self.register_buffer("d2_item_degree", self._degree_tensor(d2_inter_mat.sum(axis=0)).t())

        self.d2_restore_user_e = None
        self.d2_restore_item_e = None
        self.phase = "BOTH"

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _d2_to_union(self, d2_user_ids):
        return torch.where(
            d2_user_ids < self.n_shared_users,
            d2_user_ids,
            d2_user_ids + (self.d1_n_users - self.n_shared_users),
        )

    def _d2_to_union_np(self, user_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        return np.where(
            user_ids < self.n_shared_users,
            user_ids,
            user_ids + (self.d1_n_users - self.n_shared_users),
        )

    def _d1_item_to_total(self, item_ids):
        return item_ids + self.d1_item_offset

    def _d2_item_to_total(self, item_ids):
        return item_ids

    def _build_inter_matrix(self, rows, domain):
        rows = np.asarray(rows, dtype=np.int64)
        if domain == 1:
            users = rows[:, 0]
            items = rows[:, 1] + self.d1_item_offset
        else:
            users = self._d2_to_union_np(rows[:, 0])
            items = rows[:, 1]
        data = np.ones(len(rows), dtype=np.float32)
        return sp.coo_matrix((data, (users, items)), shape=(self.n_users, self.n_items), dtype=np.float32)

    def _build_norm_adj(self, inter_mat):
        n_users, n_items = inter_mat.shape
        adj = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_t = inter_mat.transpose()
        data_dict = dict(zip(zip(inter_mat.row, inter_mat.col + n_users), [1] * inter_mat.nnz))
        data_dict.update(dict(zip(zip(inter_t.row + n_users, inter_t.col), [1] * inter_t.nnz)))
        adj._update(data_dict)
        rowsum = np.array((adj > 0).sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(rowsum, -0.5)
        norm = sp.diags(d_inv_sqrt).dot(adj).dot(sp.diags(d_inv_sqrt)).tocoo()
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _degree_tensor(self, values):
        return torch.as_tensor(np.asarray(values), dtype=self.dtype, device=self.device)

    def get_ego_embeddings(self, domain):
        if domain == 1:
            user_emb = self.d1_user_embedding.weight
            item_emb = self.d1_item_embedding.weight
            graph = self.d1_graph
        else:
            user_emb = self.d2_user_embedding.weight
            item_emb = self.d2_item_embedding.weight
            graph = self.d2_graph
        return torch.cat([user_emb, item_emb], dim=0), graph

    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = all_embeddings + new_embeddings
        return self.dropout(new_embeddings)

    def transfer_layer(self, d1_all_embeddings, d2_all_embeddings):
        d1_user, d1_item = torch.split(d1_all_embeddings, [self.n_users, self.n_items])
        d2_user, d2_item = torch.split(d2_all_embeddings, [self.n_users, self.n_items])

        d1_user_lam = self.lambda_source * d1_user + (1 - self.lambda_source) * d2_user
        d2_user_lam = self.lambda_target * d2_user + (1 - self.lambda_target) * d1_user
        d1_item_lam = self.lambda_source * d1_item + (1 - self.lambda_source) * d2_item
        d2_item_lam = self.lambda_target * d2_item + (1 - self.lambda_target) * d1_item

        user_laplace = self.d1_user_degree + self.d2_user_degree + 1e-7
        d1_user_lap = (self.d1_user_degree * d1_user + self.d2_user_degree * d2_user) / user_laplace
        d2_user_lap = d1_user_lap

        item_laplace = self.d1_item_degree + self.d2_item_degree + 1e-7
        d1_item_lap = (self.d1_item_degree * d1_item + self.d2_item_degree * d2_item) / item_laplace
        d2_item_lap = d1_item_lap

        d1_user_specific = d1_user[self.n_shared_users:]
        d2_user_specific = d2_user[self.n_shared_users:]
        d1_user_overlap = (d1_user_lam[: self.n_shared_users] + d1_user_lap[: self.n_shared_users]) / 2
        d2_user_overlap = (d2_user_lam[: self.n_shared_users] + d2_user_lap[: self.n_shared_users]) / 2

        d1_item_specific = d1_item[self.n_shared_items:]
        d2_item_specific = d2_item[self.n_shared_items:]
        d1_item_overlap = (d1_item_lam[: self.n_shared_items] + d1_item_lap[: self.n_shared_items]) / 2
        d2_item_overlap = (d2_item_lam[: self.n_shared_items] + d2_item_lap[: self.n_shared_items]) / 2

        d1_user_transfer = torch.cat([d1_user_overlap, d1_user_specific], dim=0)
        d2_user_transfer = torch.cat([d2_user_overlap, d2_user_specific], dim=0)
        d1_item_transfer = torch.cat([d1_item_overlap, d1_item_specific], dim=0)
        d2_item_transfer = torch.cat([d2_item_overlap, d2_item_specific], dim=0)

        return (
            torch.cat([d1_user_transfer, d1_item_transfer], dim=0),
            torch.cat([d2_user_transfer, d2_item_transfer], dim=0),
        )

    def forward(self):
        d1_all, d1_graph = self.get_ego_embeddings(1)
        d2_all, d2_graph = self.get_ego_embeddings(2)

        d1_layers = [d1_all]
        d2_layers = [d2_all]
        for _ in range(self.n_layers):
            d1_all = self.graph_layer(d1_graph, d1_all)
            d2_all = self.graph_layer(d2_graph, d2_all)
            if self.phase == "BOTH" and self.is_transfer:
                d1_all, d2_all = self.transfer_layer(d1_all, d2_all)

            d1_layers.append(F.normalize(d1_all, p=2, dim=1))
            d2_layers.append(F.normalize(d2_all, p=2, dim=1))

        if self.connect_way == "concat":
            d1_emb = torch.cat(d1_layers, dim=1)
            d2_emb = torch.cat(d2_layers, dim=1)
        else:
            d1_emb = torch.stack(d1_layers, dim=1).mean(dim=1)
            d2_emb = torch.stack(d2_layers, dim=1).mean(dim=1)

        d1_user, d1_item = torch.split(d1_emb, [self.n_users, self.n_items])
        d2_user, d2_item = torch.split(d2_emb, [self.n_users, self.n_items])

        return (
            d1_user[: self.d1_n_users],
            d1_item[self.d1_item_offset:self.d1_item_offset + self.d1_n_items],
            torch.cat([d2_user[: self.n_shared_users], d2_user[self.d1_n_users:self.n_users]], dim=0),
            d2_item[: self.d2_n_items],
        )

    @staticmethod
    def _score(user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=1)

    def _emb_loss(self, *embeddings):
        if embeddings[-1].shape[0] == 0:
            return embeddings[-1].new_zeros(())
        emb_loss = embeddings[-1].new_zeros(())
        for embedding in embeddings:
            emb_loss = emb_loss + torch.norm(embedding, p=2)
        return emb_loss / embeddings[-1].shape[0]

    def _domain_loss(
        self,
        user_emb,
        item_emb,
        raw_user_table,
        raw_item_table,
        users,
        pos_items,
        neg_items,
        raw_users,
        raw_pos_items,
        raw_neg_items,
    ):
        if users.numel() == 0:
            return user_emb.new_zeros(())
        batch_user = torch.cat([users, users], dim=0)
        batch_item = torch.cat([pos_items, neg_items], dim=0)
        batch_raw_user = torch.cat([raw_users, raw_users], dim=0)
        batch_raw_item = torch.cat([raw_pos_items, raw_neg_items], dim=0)
        labels = torch.cat(
            [
                torch.ones(users.shape[0], dtype=self.dtype, device=users.device),
                torch.zeros(users.shape[0], dtype=self.dtype, device=users.device),
            ],
            dim=0,
        )

        output = self.sigmoid(self._score(user_emb[batch_user], item_emb[batch_item]))
        bce_loss = self.loss(output, labels)
        reg_loss = self._emb_loss(raw_user_table(batch_raw_user), raw_item_table(batch_raw_item))
        return bce_loss + self.reg_weight * reg_loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        self.init_restore_e()
        d1_user, d1_item, d2_user, d2_item = self.forward()

        d1_u = inter_d1[:, 0].long()
        d1_pos = inter_d1[:, 1].long()
        d1_neg = neg_d1_item.long()
        d1_loss = self._domain_loss(
            d1_user,
            d1_item,
            self.d1_user_embedding,
            self.d1_item_embedding,
            d1_u,
            d1_pos,
            d1_neg,
            d1_u,
            self._d1_item_to_total(d1_pos),
            self._d1_item_to_total(d1_neg),
        )

        d2_u = inter_d2[:, 0].long()
        d2_pos = inter_d2[:, 1].long()
        d2_neg = neg_d2_item.long()
        d2_u_total = self._d2_to_union(d2_u)
        d2_loss = self._domain_loss(
            d2_user,
            d2_item,
            self.d2_user_embedding,
            self.d2_item_embedding,
            d2_u,
            d2_pos,
            d2_neg,
            d2_u_total,
            self._d2_item_to_total(d2_pos),
            self._d2_item_to_total(d2_neg),
        )

        return (d1_loss + d2_loss) / 2

    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, history):
        def score_fn(users):
            return user_emb[users] @ item_emb.t()

        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user, d1_item, d2_user, d2_item = self.get_restore_e()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mask_d2)
        d1 = self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mask_d1)
        d2 = self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mask_d2)
        return [d1, d2]

    def init_restore_e(self):
        if self.d2_restore_user_e is not None or self.d2_restore_item_e is not None:
            self.d2_restore_user_e, self.d2_restore_item_e = None, None

    def get_restore_e(self):
        if self.d2_restore_user_e is None or self.d2_restore_item_e is None:
            d1_user, d1_item, d2_user, d2_item = self.forward()
            self.d2_restore_user_e, self.d2_restore_item_e = d2_user, d2_item
            return d1_user, d1_item, d2_user, d2_item
        d1_user, d1_item, _, _ = self.forward()
        return d1_user, d1_item, self.d2_restore_user_e, self.d2_restore_item_e

    def set_phase(self, phase):
        self.phase = phase
