import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import convert_sp_mat_to_sp_tensor, full_rank_eval, symmetric_normalize


def _proj_layer(dim, drop_rate):
    return nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop_rate))


def _mlp_layer(in_dim, out_dim, drop_rate, activation='relu'):
    act = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}.get(activation, nn.ReLU())
    return nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(drop_rate))


class DGCDR(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params.get("embedding_size", 64)
        self.n_layers = params.get("dgcdr_n_layers", 3)
        self.drop_rate = params.get("dgcdr_drop_rate", 0.3)
        self.reg_weight = params.get("dgcdr_reg_weight", 1e-4)
        self.connect_way = "concat"
        self.tau = params.get("dgcdr_tau", 0.15)

        self.preference_disentangle = True
        self.fuse_mode = "attention"
        self.attention_mode = "all"
        self.concat_mode = "part"
        self.feature_mapping_way = "projection"
        self.item_disentangle = True
        self.item_mapping = False
        self.item_negative = True

        self.cl_sim_weight = params.get("dgcdr_cl_sim_weight", 0.05)
        self.cl_org_weight = params.get("dgcdr_cl_org_weight", 0.5)
        self.cl_decoder_weight = params.get("dgcdr_cl_decoder_weight", 0.1)
        self.item_cl_weight = params.get("dgcdr_item_cl_weight", 0.1)

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

        self.d1_graph = self._build_norm_graph(args.d1["inter_mat"])
        self.d2_graph = self._build_norm_graph(args.d2["inter_mat"])

        if self.connect_way == 'concat':
            en_dim = self.emb_dim * (self.n_layers + 1)
        else:
            en_dim = self.emb_dim

        if self.preference_disentangle:
            if self.feature_mapping_way == 'projection':
                self.d1_en_common = _proj_layer(en_dim, self.drop_rate)
                self.d1_en_specific = _proj_layer(en_dim, self.drop_rate)
                self.d2_en_common = _proj_layer(en_dim, self.drop_rate)
                self.d2_en_specific = _proj_layer(en_dim, self.drop_rate)
                self.d1_de = _proj_layer(en_dim, self.drop_rate)
                self.d2_de = _proj_layer(en_dim, self.drop_rate)
                self.d1_en_item_common = _proj_layer(en_dim, self.drop_rate)
                self.d1_en_item_specific = _proj_layer(en_dim, self.drop_rate)
                self.d2_en_item_common = _proj_layer(en_dim, self.drop_rate)
                self.d2_en_item_specific = _proj_layer(en_dim, self.drop_rate)
            elif self.feature_mapping_way == 'mlp':
                mlp_dim = en_dim
                act = "relu"
                self.d1_en = _mlp_layer(en_dim, mlp_dim, self.drop_rate, act)
                self.d1_de = _mlp_layer(mlp_dim, en_dim, self.drop_rate, act)
                self.d2_en = _mlp_layer(en_dim, mlp_dim, self.drop_rate, act)
                self.d2_de = _mlp_layer(mlp_dim, en_dim, self.drop_rate, act)
                self.d1_en_item = _mlp_layer(en_dim, mlp_dim, self.drop_rate, act)
                self.d2_en_item = _mlp_layer(en_dim, mlp_dim, self.drop_rate, act)

            if self.fuse_mode == 'concat':
                if self.concat_mode == 'all':
                    map_in = en_dim + 2 * en_dim
                else:
                    map_in = 2 * en_dim
                self.mapping = _mlp_layer(map_in, en_dim, self.drop_rate, "relu")

            if (not self.item_disentangle) and self.item_mapping:
                self.d1_item_map = _mlp_layer(en_dim, en_dim, self.drop_rate, "relu")
                self.d2_item_map = _mlp_layer(en_dim, en_dim, self.drop_rate, "relu")

            self.sim_loss = nn.CosineEmbeddingLoss(margin=0.5)

        self.neg_ratio = params.get("neg_ratio", 1)

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

    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[: self.d1_n_users], emb[self.n_users:]
        user = torch.cat(
            [emb[: self.n_shared_users], emb[self.d1_n_users: self.n_users]], dim=0
        )
        return user, emb[self.n_users:]


    def graph_layer(self, adj, all_emb):
        side = torch.sparse.mm(adj, all_emb)
        out = side + torch.mul(all_emb, side)
        out = all_emb + out
        return self.dropout(out)


    def fuse_and_update(self, common, specific, gnn_emb):
        if self.fuse_mode == 'concat':
            if self.concat_mode == 'all':
                return self.mapping(torch.cat((gnn_emb, common, specific), -1))
            else:
                return self.mapping(torch.cat((common, specific), -1))
        elif self.fuse_mode == 'attention':
            a_c = torch.sum(torch.mul(gnn_emb, common), dim=1, keepdim=True)
            a_s = torch.sum(torch.mul(gnn_emb, specific), dim=1, keepdim=True)
            att = F.softmax(torch.cat((a_c, a_s), dim=1), dim=1)
            e_c = att[:, 0:1] * common
            e_s = att[:, 1:2] * specific
            if self.attention_mode == 'all':
                return gnn_emb + e_c + e_s
            else:
                return e_c + e_s


    def disentangle_layer(self, d1_emb, d2_emb, is_user=True):
        if is_user:
            ov = self.n_shared_users
            d1_indep = d1_emb[ov:]
            d2_indep = d2_emb[ov:]
            d1_ov = d1_emb[:ov]
            d2_ov = d2_emb[:ov]

            if self.feature_mapping_way == 'projection':
                d1_c = d1_ov * torch.sigmoid(self.d1_en_common(d1_ov))
                d1_s = d1_ov * torch.sigmoid(self.d1_en_specific(d1_ov))
                d2_c = d2_ov * torch.sigmoid(self.d2_en_common(d2_ov))
                d2_s = d2_ov * torch.sigmoid(self.d2_en_specific(d2_ov))
                d1_de_user = d1_ov * torch.sigmoid(self.d1_de(d1_ov))
                d1_de_c = d1_c * torch.sigmoid(self.d1_de(d1_c))
                d1_de_s = d1_s * torch.sigmoid(self.d1_de(d1_s))
                d2_de_user = d2_ov * torch.sigmoid(self.d2_de(d2_ov))
                d2_de_c = d2_c * torch.sigmoid(self.d2_de(d2_c))
                d2_de_s = d2_s * torch.sigmoid(self.d2_de(d2_s))
            elif self.feature_mapping_way == 'mlp':
                d1_c = self.d1_en(d1_ov)
                d1_s = d1_ov - d1_c
                d2_c = self.d2_en(d2_ov)
                d2_s = d2_ov - d2_c
                d1_de_user = self.d1_de(d1_ov)
                d1_de_c = self.d1_de(d1_c)
                d1_de_s = self.d1_de(d1_s)
                d2_de_user = self.d2_de(d2_ov)
                d2_de_c = self.d2_de(d2_c)
                d2_de_s = self.d2_de(d2_s)

            d1_fused = self.fuse_and_update(d1_c, d1_s, d1_ov)
            d2_fused = self.fuse_and_update(d2_c, d2_s, d2_ov)

            d1_all_user = torch.cat([d1_fused, d1_indep], dim=0)
            d2_all_user = torch.cat([d2_fused, d2_indep], dim=0)

            disentangled = [d1_c, d2_c, d1_s, d2_s, d1_ov, d2_ov,
                            d1_de_user, d1_de_c, d1_de_s,
                            d2_de_user, d2_de_c, d2_de_s]
            return d1_all_user, d2_all_user, disentangled
        else:
            if self.feature_mapping_way == 'projection':
                d1_c = d1_emb * torch.sigmoid(self.d1_en_item_common(d1_emb))
                d1_s = d1_emb * torch.sigmoid(self.d1_en_item_specific(d1_emb))
                d2_c = d2_emb * torch.sigmoid(self.d2_en_item_common(d2_emb))
                d2_s = d2_emb * torch.sigmoid(self.d2_en_item_specific(d2_emb))
            elif self.feature_mapping_way == 'mlp':
                d1_c = self.d1_en_item(d1_emb)
                d1_s = d1_emb - d1_c
                d2_c = self.d2_en_item(d2_emb)
                d2_s = d2_emb - d2_c
            d1_fused = self.fuse_and_update(d1_c, d1_s, d1_emb)
            d2_fused = self.fuse_and_update(d2_c, d2_s, d2_emb)
            return d1_fused, d2_fused


    def forward(self):
        d1_all = torch.cat([self.d1_user.weight, self.d1_item.weight])
        d1_layers = [d1_all]
        for _ in range(self.n_layers):
            d1_all = self.graph_layer(self.d1_graph, d1_all)
            d1_layers.append(d1_all)

        d2_all = torch.cat([self.d2_user.weight, self.d2_item.weight])
        d2_layers = [d2_all]
        for _ in range(self.n_layers):
            d2_all = self.graph_layer(self.d2_graph, d2_all)
            d2_layers.append(d2_all)

        if self.connect_way == "mean":
            d1_emb = torch.stack(d1_layers, dim=1).mean(dim=1)
            d2_emb = torch.stack(d2_layers, dim=1).mean(dim=1)
        elif self.connect_way == "concat":
            d1_emb = torch.cat(d1_layers, dim=1)
            d2_emb = torch.cat(d2_layers, dim=1)
        else:
            raise ValueError(f"Unsupported connect_way: {self.connect_way}")

        d1_user, d1_item = self._split_domain_ui(d1_emb, 1)
        d2_user, d2_item = self._split_domain_ui(d2_emb, 2)

        if self.preference_disentangle:
            d1_user, d2_user, user_dis_list = self.disentangle_layer(d1_user, d2_user, is_user=True)
            if self.item_disentangle:
                d1_item, d2_item = self.disentangle_layer(d1_item, d2_item, is_user=False)
            elif self.item_mapping:
                d1_item = d1_item * torch.sigmoid(self.d1_item_map(d1_item))
                d2_item = d2_item * torch.sigmoid(self.d2_item_map(d2_item))
        else:
            user_dis_list = []

        return user_dis_list, d1_user, d1_item, d2_user, d2_item


    def decoder_loss_function(self, anchor, de_user, de_c, de_s, t):
        anchor = F.normalize(anchor, dim=1)
        de_user = F.normalize(de_user, dim=1)
        de_c = F.normalize(de_c, dim=1)
        de_s = F.normalize(de_s, dim=1)
        pos_1 = torch.sum(anchor * de_c, dim=1)
        pos_2 = torch.sum(anchor * de_user, dim=1)
        neg_1 = torch.sum(anchor * de_s, dim=1)
        p1 = torch.exp(pos_1 / t)
        p2 = torch.exp(pos_2 / t)
        n1 = torch.exp(neg_1 / t)
        loss_cg = -torch.mean(torch.log(p1 / (p1 + p2) + 1e-24))
        loss_gs = -torch.mean(torch.log(p2 / (p2 + n1) + 1e-24))
        return loss_cg + loss_gs

    def item_disentangle_loss(self, dom_s, cross_s, item_e, t):
        dom_s = F.normalize(dom_s, dim=1)
        cross_s = F.normalize(cross_s, dim=1)
        item_e = F.normalize(item_e, dim=1)
        pos = torch.sum(dom_s * item_e, dim=1)
        neg = torch.sum(cross_s * item_e, dim=1)
        pos_h = torch.exp(pos / t)
        neg_h = torch.exp(neg / t)
        return -torch.mean(torch.log(pos_h / (pos_h + neg_h) + 1e-24))


    def get_score(self, user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)

    def get_bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos = self.get_score(user_emb, pos_item_emb)
        neg = self.get_score(user_emb, neg_item_emb)
        return -F.logsigmoid(pos - neg).mean()

    def get_reg_loss(self, *embs):
        loss = sum(e.norm(2).pow(2) for e in embs)
        return loss / (2 * embs[0].shape[0])


    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        user_dis_list, d1_user, d1_item, d2_user, d2_item = self.forward()

        d1_users_idx = inter_d1[:, 0]
        d1_items_idx = inter_d1[:, 1]
        d2_users_idx = inter_d2[:, 0]
        d2_items_idx = inter_d2[:, 1]

        losses = []
        d1_u = d1_user[d1_users_idx]
        d1_pos = d1_item[d1_items_idx]
        d2_u = d2_user[d2_users_idx]
        d2_pos = d2_item[d2_items_idx]

        d1_neg = d1_item[neg_d1_item]
        d1_loss = self.get_bpr_loss(d1_u, d1_pos, d1_neg)
        d1_loss += self.reg_weight * self.get_reg_loss(
            self.d1_user.weight[d1_users_idx],
            self.d1_item.weight[d1_items_idx],
            self.d1_item.weight[neg_d1_item]
        )
        d2_neg = d2_item[neg_d2_item]
        d2_loss = self.get_bpr_loss(d2_u, d2_pos, d2_neg)
        d2_loss += self.reg_weight * self.get_reg_loss(
            self.d2_user.weight[d2_users_idx],
            self.d2_item.weight[d2_items_idx],
            self.d2_item.weight[neg_d2_item]
        )
        losses.append(d1_loss)
        losses.append(d2_loss)

        if self.preference_disentangle and len(user_dis_list) > 0:
            d1_c, d2_c, d1_s, d2_s, d1_ov, d2_ov, \
                d1_de_user, d1_de_c, d1_de_s, d2_de_user, d2_de_c, d2_de_s = user_dis_list

            ov = self.n_shared_users

            is_d1_shared = (d1_users_idx >= 0) & (d1_users_idx < ov)
            d1_sh_u = d1_users_idx[is_d1_shared]
            is_d2_shared = (d2_users_idx >= 0) & (d2_users_idx < ov)
            d2_sh_u = d2_users_idx[is_d2_shared]

            if d1_sh_u.numel() > 0 and d2_sh_u.numel() > 0:
                d1_sh_c = d1_c[d1_sh_u]
                d1_cross_c = d2_c[d1_sh_u]
                d2_sh_c = d2_c[d2_sh_u]
                d2_cross_c = d1_c[d2_sh_u]

                L_sim_d1 = self.sim_loss(d1_sh_c, d1_cross_c, torch.ones(d1_sh_c.size(0), device=self.device))
                L_sim_d2 = self.sim_loss(d2_sh_c, d2_cross_c, torch.ones(d2_sh_c.size(0), device=self.device))
                losses.append(self.cl_sim_weight * (L_sim_d1 + L_sim_d2))

                if self.cl_org_weight != 0:
                    d1_sh_s = d1_s[d1_sh_u]
                    d2_sh_s = d2_s[d2_sh_u]
                    L_ort_d1 = torch.mean(torch.sum(torch.mul(d1_sh_c, d1_sh_s), dim=1) ** 2)
                    L_ort_d2 = torch.mean(torch.sum(torch.mul(d2_sh_c, d2_sh_s), dim=1) ** 2)
                    losses.append(self.cl_org_weight * L_ort_d1)
                    losses.append(self.cl_org_weight * L_ort_d2)

                if self.cl_decoder_weight != 0:
                    d1_de_u_sh = d1_de_user[d2_sh_u]
                    d1_de_c_sh = d1_de_c[d2_sh_u]
                    d1_de_s_sh = d1_de_s[d2_sh_u]
                    d2_de_u_sh = d2_de_user[d1_sh_u]
                    d2_de_c_sh = d2_de_c[d1_sh_u]
                    d2_de_s_sh = d2_de_s[d1_sh_u]

                    d1_anchor = d1_ov[d1_sh_u]
                    d2_anchor = d2_ov[d2_sh_u]

                    loss_T2S = self.decoder_loss_function(d1_anchor, d2_de_u_sh, d2_de_c_sh, d2_de_s_sh, self.tau)
                    loss_S2T = self.decoder_loss_function(d2_anchor, d1_de_u_sh, d1_de_c_sh, d1_de_s_sh, self.tau)
                    losses.append(self.cl_decoder_weight * loss_T2S)
                    losses.append(self.cl_decoder_weight * loss_S2T)

                if self.item_negative:
                    if d1_sh_u.numel() > 0:
                        d1_sh_s_feat = d1_s[d1_sh_u]
                        d1_cross_s_feat = d2_s[d1_sh_u]
                        d1_sh_items = d1_items_idx[is_d1_shared]
                        d1_sh_item_e = d1_item[d1_sh_items]
                        losses.append(self.item_cl_weight * self.item_disentangle_loss(
                            d1_sh_s_feat, d1_cross_s_feat, d1_sh_item_e, self.tau))

                    if d2_sh_u.numel() > 0:
                        d2_sh_s_feat = d2_s[d2_sh_u]
                        d2_cross_s_feat = d1_s[d2_sh_u]
                        d2_sh_items = d2_items_idx[is_d2_shared]
                        d2_sh_item_e = d2_item[d2_sh_items]
                        losses.append(self.item_cl_weight * self.item_disentangle_loss(
                            d2_sh_s_feat, d2_cross_s_feat, d2_sh_item_e, self.tau))

        return sum(losses)


    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        _, d1_user, d1_item, d2_user, d2_item = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        d1 = self._eval_domain(eval_set_1, d1_user, d1_item, self.d1_domain_name, mode, mask_d1)
        d2 = self._eval_domain(eval_set_2, d2_user, d2_item, self.d2_domain_name, mode, mask_d2)
        return [d1, d2]
