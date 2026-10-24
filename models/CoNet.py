import torch
import torch.nn as nn

from utils import full_rank_eval


def _xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class CoNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.overlapped_num_users = int(args.data["n_shared_users"])
        self.source_num_users = int(args.d1["n_users"])
        self.target_num_users = int(args.d2["n_users"])
        self.source_num_items = int(args.d1["n_items"])
        self.target_num_items = int(args.d2["n_items"])
        self.total_num_users = self.source_num_users + self.target_num_users - self.overlapped_num_users
        self.total_num_items = self.source_num_items + self.target_num_items
        self.overlapped_num_items = 0

        self.source_domain_name = args.d1.get("name", "domain1").lower()
        self.target_domain_name = args.d2.get("name", "domain2").lower()
        self.target_only_user_count = self.target_num_users - self.overlapped_num_users
        self.source_item_offset = self.target_num_items

        if self.overlapped_num_users > 0:
            self.mode = "overlap_users"
        elif self.overlapped_num_items > 0:
            self.mode = "overlap_items"
        else:
            self.mode = "non_overlap"

        self.latent_dim = int(params.get("conet_embedding_size", 64))
        self.reg_weight = float(params.get("conet_reg_weight", params.get("reg_weight", 0.01)))
        self.cross_layers = list(params.get("conet_mlp_hidden_size", [64, 32]))
        self.is_transfer = bool(params.get("conet_is_transfer", True))
        self.eval_user_chunk_size = int(params.get("conet_eval_user_chunk_size", 64))
        self.eval_item_chunk_size = int(params.get("conet_eval_item_chunk_size", 2048))

        self.source_user_embedding = nn.Embedding(self.total_num_users, self.latent_dim, dtype=self.dtype)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.latent_dim, dtype=self.dtype)
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.latent_dim, dtype=self.dtype)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.latent_dim, dtype=self.dtype)

        self.loss = nn.BCELoss()

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users:self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items:self.target_num_items].fill_(0)
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.source_crossunit_linear, self.source_crossunit_act = self.cross_units(
            [2 * self.latent_dim] + self.cross_layers
        )
        self.source_outputunit = nn.Sequential(nn.Linear(self.cross_layers[-1], 1), nn.Sigmoid())

        self.target_crossunit_linear, self.target_crossunit_act = self.cross_units(
            [2 * self.latent_dim] + self.cross_layers
        )
        self.target_outputunit = nn.Sequential(nn.Linear(self.cross_layers[-1], 1), nn.Sigmoid())

        self.crossparas = self.cross_parameters([2 * self.latent_dim] + self.cross_layers)
        self.apply(_xavier_normal_initialization)

    def _d1_user_to_total(self, user):
        return torch.where(
            user < self.overlapped_num_users,
            user,
            user + self.target_only_user_count,
        )

    def _d2_user_to_total(self, user):
        return user

    def _d1_item_to_total(self, item):
        return item + self.source_item_offset

    def _d2_item_to_total(self, item):
        return item

    def cross_units(self, cross_layers):
        cross_modules_linear, cross_modules_act = [], []
        for d_in, d_out in zip(cross_layers[:-1], cross_layers[1:]):
            cross_modules_linear.append(nn.Linear(d_in, d_out))
            cross_modules_act.append(nn.ReLU())
        return nn.ModuleList(cross_modules_linear), nn.ModuleList(cross_modules_act)

    def cross_parameters(self, cross_layers):
        cross_paras = []
        for d_in, d_out in zip(cross_layers[:-1], cross_layers[1:]):
            cross_paras.append(nn.Linear(d_in, d_out, bias=False))
        return nn.ModuleList(cross_paras)

    def source_forward(self, user, item):
        source_user_embedding = self.source_user_embedding(user)
        source_item_embedding = self.source_item_embedding(item)
        target_user_embedding = self.target_user_embedding(user)
        target_item_embedding = self.target_item_embedding(item)
        source_crossinput = torch.cat([source_user_embedding, source_item_embedding], dim=1).to(self.device)
        target_crossinput = torch.cat([target_user_embedding, target_item_embedding], dim=1).to(self.device)

        if self.mode == "overlap_users":
            overlap_idx = user < self.overlapped_num_users
        else:
            overlap_idx = item < self.overlapped_num_items

        for i in range(len(self.source_crossunit_linear)):
            source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
            cross_para = self.crossparas[i].weight.t()
            target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]

            source_crossoutput = source_fc_module(source_crossinput)
            source_crossoutput[overlap_idx] = source_crossoutput[overlap_idx] + torch.mm(target_crossinput, cross_para)[
                overlap_idx
            ]
            source_crossoutput = source_act_module(source_crossoutput)

            target_crossoutput = target_fc_module(target_crossinput)
            target_crossoutput[overlap_idx] = target_crossoutput[overlap_idx] + torch.mm(source_crossinput, cross_para)[
                overlap_idx
            ]
            target_crossoutput = target_act_module(target_crossoutput)

            source_crossinput = source_crossoutput
            target_crossinput = target_crossoutput

        return self.source_outputunit(source_crossinput).squeeze()

    def target_forward(self, user, item):
        source_user_embedding = self.source_user_embedding(user)
        source_item_embedding = self.source_item_embedding(item)
        target_user_embedding = self.target_user_embedding(user)
        target_item_embedding = self.target_item_embedding(item)
        source_crossinput = torch.cat([source_user_embedding, source_item_embedding], dim=1).to(self.device)
        target_crossinput = torch.cat([target_user_embedding, target_item_embedding], dim=1).to(self.device)

        if self.mode == "overlap_users":
            overlap_idx = user < self.overlapped_num_users
        else:
            overlap_idx = item < self.overlapped_num_items

        for i in range(len(self.target_crossunit_linear)):
            source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
            cross_para = self.crossparas[i].weight.t()
            target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]

            source_crossoutput = source_fc_module(source_crossinput)
            source_crossoutput[overlap_idx] = source_crossoutput[overlap_idx] + torch.mm(target_crossinput, cross_para)[
                overlap_idx
            ]
            source_crossoutput = source_act_module(source_crossoutput)

            target_crossoutput = target_fc_module(target_crossinput)
            target_crossoutput[overlap_idx] = target_crossoutput[overlap_idx] + torch.mm(source_crossinput, cross_para)[
                overlap_idx
            ]
            target_crossoutput = target_act_module(target_crossoutput)

            source_crossinput = source_crossoutput
            target_crossinput = target_crossoutput

        return self.target_outputunit(target_crossinput).squeeze()

    def _plain_forward_from_emb(self, user_e, item_e, domain):
        inputs = torch.cat([user_e, item_e], dim=1).to(self.device)
        if domain == 1:
            linears, acts, output = self.source_crossunit_linear, self.source_crossunit_act, self.source_outputunit
        else:
            linears, acts, output = self.target_crossunit_linear, self.target_crossunit_act, self.target_outputunit

        for i in range(len(linears)):
            inputs = acts[i](linears[i](inputs))
        return output(inputs).view(-1)

    def target_forward_without_cross(self, user, item):
        target_user_embedding = self.target_user_embedding(user)
        target_item_embedding = self.target_item_embedding(item)
        return self._plain_forward_from_emb(target_user_embedding, target_item_embedding, domain=2)

    def source_forward_without_cross(self, user, item):
        source_user_embedding = self.source_user_embedding(user)
        source_item_embedding = self.source_item_embedding(item)
        return self._plain_forward_from_emb(source_user_embedding, source_item_embedding, domain=1)

    def _pointwise_bce(self, forward_fn, user, pos_item, neg_item):
        if user.numel() == 0:
            return self.source_user_embedding.weight.new_zeros(())
        p_pos = forward_fn(user, pos_item).reshape(-1)
        p_neg = forward_fn(user, neg_item).reshape(-1)
        pred = torch.cat([p_pos, p_neg], dim=0)
        label = torch.cat([torch.ones_like(p_pos), torch.zeros_like(p_neg)], dim=0)
        return self.loss(pred, label)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        source_user = self._d1_user_to_total(inter_d1[:, 0].long())
        source_item = self._d1_item_to_total(inter_d1[:, 1].long())
        source_neg_item = self._d1_item_to_total(neg_d1_item.long())

        target_user = self._d2_user_to_total(inter_d2[:, 0].long())
        target_item = self._d2_item_to_total(inter_d2[:, 1].long())
        target_neg_item = self._d2_item_to_total(neg_d2_item.long())

        if self.is_transfer:
            loss_s = self._pointwise_bce(self.source_forward, source_user, source_item, source_neg_item)
            loss_t = self._pointwise_bce(self.target_forward, target_user, target_item, target_neg_item)
            reg_loss = self.source_user_embedding.weight.new_zeros(())
            for para in self.crossparas:
                reg_loss = reg_loss + torch.norm(para.weight)
            return loss_s + loss_t + self.reg_weight * reg_loss

        return self._pointwise_bce(self.target_forward_without_cross, target_user, target_item, target_neg_item)

    def _full_sort_scores(self, users, domain):
        if domain == 1:
            total_users = self._d1_user_to_total(users.long())
            user_e = self.source_user_embedding(total_users)
            item_ids = self._d1_item_to_total(
                torch.arange(self.source_num_items, device=users.device, dtype=torch.long)
            )
            item_e = self.source_item_embedding(item_ids)
        else:
            total_users = self._d2_user_to_total(users.long())
            user_e = self.target_user_embedding(total_users)
            item_ids = self._d2_item_to_total(
                torch.arange(self.target_num_items, device=users.device, dtype=torch.long)
            )
            item_e = self.target_item_embedding(item_ids)

        scores = user_e.new_empty((user_e.size(0), item_e.size(0)))
        user_step = self.eval_user_chunk_size if self.eval_user_chunk_size > 0 else user_e.size(0)
        item_step = self.eval_item_chunk_size if self.eval_item_chunk_size > 0 else item_e.size(0)

        for u_start in range(0, user_e.size(0), user_step):
            u_end = min(u_start + user_step, user_e.size(0))
            u_chunk = user_e[u_start:u_end]
            for i_start in range(0, item_e.size(0), item_step):
                i_end = min(i_start + item_step, item_e.size(0))
                i_chunk = item_e[i_start:i_end]
                u_expand = u_chunk[:, None, :].expand(-1, i_chunk.size(0), -1).reshape(-1, self.latent_dim)
                i_expand = i_chunk[None, :, :].expand(u_chunk.size(0), -1, -1).reshape(-1, self.latent_dim)
                scores[u_start:u_end, i_start:i_end] = self._plain_forward_from_emb(
                    u_expand, i_expand, domain
                ).view(u_chunk.size(0), i_chunk.size(0))
        return scores

    def _eval_domain(self, eval_set, domain, domain_name, mode, history):
        def score_fn(users):
            return self._full_sort_scores(users, domain)

        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 1, self.source_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, 2, self.target_domain_name, mode, mask_d2)
        d1 = self._eval_domain(eval_set_1, 1, self.source_domain_name, mode, mask_d1)
        d2 = self._eval_domain(eval_set_2, 2, self.target_domain_name, mode, mask_d2)
        return [d1, d2]
