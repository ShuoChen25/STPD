import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from utils import (
    convert_sp_mat_to_sp_tensor,
    full_rank_eval,
    load_pca_item_initializers,
    symmetric_normalize,
)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def _grad_reverse(x, lambd=1.0):
    return _GradientReverse.apply(x, lambd)


class TagGraph(nn.Module):

    @staticmethod
    def _gate_logits_init(n_edges, freqs, base, gamma, freq_init):
        if not n_edges:
            return torch.zeros(0, dtype=torch.float32)
        if freq_init and freqs is not None and freqs.numel() > 0:
            freq_score = torch.log1p(freqs.to(dtype=torch.float32))
            freq_score = freq_score / freq_score.max().clamp(min=1.0)
            return base + gamma * freq_score
        return torch.full((n_edges,), float(base), dtype=torch.float32)

    @staticmethod
    def _sparsity_init(n_edges, weights, name):
        if not n_edges:
            return torch.zeros(0, dtype=torch.float32)
        if weights is None:
            return torch.ones(n_edges, dtype=torch.float32)
        sw = weights.to(dtype=torch.float32).reshape(-1)
        assert sw.numel() == n_edges, (
            f"{name} length {sw.numel()} != n_edges {n_edges}"
        )
        return sw

    @staticmethod
    def _undirected(edges, n_edges, src_offset, dst_offset):
        if not n_edges:
            z = torch.zeros(0, dtype=torch.long)
            return z, z, z
        src = edges[:, 0].long() + src_offset
        dst = edges[:, 1].long() + dst_offset
        eidx = torch.arange(n_edges, dtype=torch.long)
        return (torch.cat([src, dst]), torch.cat([dst, src]),
                torch.cat([eidx, eidx]))

    def __init__(self, n_anchors, n_tag1, n_tag2,
                 at1_edges, at1_freqs, t1t2_edges,
                 dtype, base=-2.0, gamma=2.0, freq_init=True,
                 at1_sparsity_weights=None,
                 at2_edges=None, at2_freqs=None, at2_sparsity_weights=None,
                 topology="chain"):
        super().__init__()
        if topology not in ("chain", "parallel", "swapped"):
            raise ValueError(
                "topology must be 'chain', 'parallel', or 'swapped', got "
                f"{topology!r}."
            )
        if topology in ("parallel", "swapped") and at2_edges is None:
            raise ValueError(
                f"topology={topology!r} needs at2_edges (anchor->Tag2 pairs)."
            )
        self.n_anchors = n_anchors
        self.n_tag1 = n_tag1
        self.n_tag2 = n_tag2
        self.n_total = n_anchors + n_tag1 + n_tag2
        self.topology = topology
        at1_edges = at1_edges.long()
        if topology == "swapped":
            at1_edges = at1_edges.new_zeros((0, 2))

        n_at = int(at1_edges.shape[0])
        self.at1_logits = nn.Parameter(
            self._gate_logits_init(n_at, at1_freqs, base, gamma, freq_init)
            .to(dtype=dtype)
        )
        self.register_buffer(
            "at1_sparsity_weights",
            self._sparsity_init(n_at, at1_sparsity_weights,
                                "at1_sparsity_weights").to(dtype=dtype),
            persistent=False,
        )
        self.register_buffer("at1_edges", at1_edges.clone(), persistent=False)
        at_row, at_col, at_eidx = self._undirected(at1_edges, n_at, 0, n_anchors)

        if topology in ("parallel", "swapped"):
            at2_edges = at2_edges.long()
            n_at2 = int(at2_edges.shape[0])
            at2_logits_init = self._gate_logits_init(
                n_at2, at2_freqs, base, gamma, freq_init,
            )
            at2_sw = self._sparsity_init(
                n_at2, at2_sparsity_weights, "at2_sparsity_weights",
            )
            at2_row, at2_col, at2_eidx = self._undirected(
                at2_edges, n_at2, 0, n_anchors + n_tag1,
            )
        else:
            at2_edges = torch.zeros((0, 2), dtype=torch.long)
            n_at2 = 0
            at2_logits_init = torch.zeros(0, dtype=torch.float32)
            at2_sw = torch.zeros(0, dtype=torch.float32)
            z = torch.zeros(0, dtype=torch.long)
            at2_row, at2_col, at2_eidx = z, z, z
        self.at2_logits = nn.Parameter(at2_logits_init.to(dtype=dtype))
        self.register_buffer(
            "at2_sparsity_weights", at2_sw.to(dtype=dtype), persistent=False,
        )
        self.register_buffer("at2_edges", at2_edges.clone(), persistent=False)
        self.n_at2 = n_at2

        if topology in ("chain", "swapped"):
            t1_node = t1t2_edges[:, 0].long() + n_anchors
            t2_node = t1t2_edges[:, 1].long() + n_anchors + n_tag1
            t_row = torch.cat([t1_node, t2_node])
            t_col = torch.cat([t2_node, t1_node])
        else:
            t_row = torch.zeros(0, dtype=torch.long)
            t_col = torch.zeros(0, dtype=torch.long)


        deg_static = torch.zeros(self.n_total, dtype=torch.float32)
        deg_static.scatter_add_(0, t_row, torch.ones_like(t_row, dtype=torch.float32))


        self.register_buffer(
            "t1t2_raw", t1t2_edges.long().clone(), persistent=False,
        )

        self.register_buffer("at_row", at_row, persistent=False)
        self.register_buffer("at_col", at_col, persistent=False)
        self.register_buffer("at_eidx", at_eidx, persistent=False)
        self.register_buffer("at2_row", at2_row, persistent=False)
        self.register_buffer("at2_col", at2_col, persistent=False)
        self.register_buffer("at2_eidx", at2_eidx, persistent=False)
        self.register_buffer("t_row", t_row, persistent=False)
        self.register_buffer("t_col", t_col, persistent=False)
        self.register_buffer(
            "row", torch.cat([at_row, at2_row, t_row]), persistent=False,
        )
        self.register_buffer(
            "col", torch.cat([at_col, at2_col, t_col]), persistent=False,
        )
        self.register_buffer("deg_static", deg_static.to(dtype=dtype), persistent=False)
        self.n_at = n_at

    def _edge_values(self):

        deg = self.deg_static.clone()
        gates = gates2 = None
        if self.n_at:
            gates = torch.sigmoid(self.at1_logits)
            deg = deg.scatter_add(0, self.at_row, gates[self.at_eidx])
        if self.n_at2:
            gates2 = torch.sigmoid(self.at2_logits)
            deg = deg.scatter_add(0, self.at2_row, gates2[self.at2_eidx])
        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)

        def gated(g, eidx, row, col):
            return g[eidx] * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        at_values = (
            gated(gates, self.at_eidx, self.at_row, self.at_col)
            if self.n_at else deg_inv_sqrt.new_zeros(0)
        )
        at2_values = (
            gated(gates2, self.at2_eidx, self.at2_row, self.at2_col)
            if self.n_at2 else deg_inv_sqrt.new_zeros(0)
        )
        t_values = deg_inv_sqrt[self.t_row] * deg_inv_sqrt[self.t_col]
        return torch.cat([at_values, at2_values, t_values], dim=0)

    def _run_layers(self, x, values, n_layers, dropout):
        embeddings = [x]
        cur = x
        for _ in range(n_layers):
            msg = cur[self.col] * values.unsqueeze(-1)
            out = torch.zeros_like(cur).index_add(0, self.row, msg)
            cur = dropout(out)
            embeddings.append(cur)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    def propagate(self, anchor_emb, tag1_emb, tag2_emb, n_layers, dropout):
        all_emb = torch.cat([anchor_emb, tag1_emb, tag2_emb], dim=0)
        values = self._edge_values()
        agg = self._run_layers(all_emb, values, n_layers, dropout)
        return agg[:self.n_anchors]

    def propagate_decomposed(self, anchor_emb, tag1_emb, tag2_emb,
                             n_layers, dropout):

        d = anchor_emb.shape[1]
        zeros_a = anchor_emb.new_zeros(self.n_anchors, d)
        zeros_1 = tag1_emb.new_zeros(self.n_tag1, d)
        zeros_2 = tag2_emb.new_zeros(self.n_tag2, d)

        anchor_block = torch.cat([anchor_emb, zeros_a, zeros_a], dim=1)
        tag1_block = torch.cat([zeros_1, tag1_emb, zeros_1], dim=1)
        tag2_block = torch.cat([zeros_2, zeros_2, tag2_emb], dim=1)
        x = torch.cat([anchor_block, tag1_block, tag2_block], dim=0)

        values = self._edge_values()
        agg = self._run_layers(x, values, n_layers, dropout)[:self.n_anchors]
        return agg[:, :d], agg[:, d:2 * d], agg[:, 2 * d:]

    def sparsity_loss(self):

        terms = []
        if self.n_at:
            terms.append(
                (torch.sigmoid(self.at1_logits) * self.at1_sparsity_weights).mean()
            )
        if self.n_at2:
            terms.append(
                (torch.sigmoid(self.at2_logits) * self.at2_sparsity_weights).mean()
            )
        if not terms:
            return self.at1_logits.new_zeros(())
        return torch.stack(terms).mean()


class STPD(nn.Module):
    use_tag_outputs_as_ui_input = True

    def __init__(self, args):
        super().__init__()

        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        self.emb_dim = params["embedding_size"]
        self.drop_rate = params["drop_rate"]
        self.intra_domain_layers = params["intra_domain_layers"]

        self.ui_graph_layers = int(
            params.get("ui_graph_layers", self.intra_domain_layers)
        )
        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users, self.d2_n_users = args.d1["n_users"], args.d2["n_users"]
        self.d1_n_items, self.d2_n_items = args.d1["n_items"], args.d2["n_items"]
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)


        self.d1_user_tg = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_user_tg = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_tg = nn.Embedding(
            self.d1_n_items, self.emb_dim, dtype=self.dtype
        )
        self.d2_item_tg = nn.Embedding(
            self.d2_n_items, self.emb_dim, dtype=self.dtype
        )

        self.dropout = nn.Dropout(self.drop_rate)

        self.gate_d1 = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.gate_d2 = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        self.fuse_d1 = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.fuse_d2 = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        domain_disc_hidden_dim = int(
            params.get("domain_disc_hidden_dim", self.emb_dim) or self.emb_dim
        )
        self.shared_domain_discriminator = self._make_domain_discriminator(
            domain_disc_hidden_dim,
        )
        self.private_domain_discriminator = self._make_domain_discriminator(
            domain_disc_hidden_dim,
        )

        self.d1_graph = self._build_norm_graph(args.d1["inter_mat"])
        self.d2_graph = self._build_norm_graph(args.d2["inter_mat"])

        self._build_overlap_pos_lookup(args)


        self._build_all_pos_lookup(args)

        self.apply(self._init_weights)


        self._maybe_load_pca_item_init(args)


        nn.init.xavier_uniform_(self.d1_item.weight, gain=1)
        nn.init.xavier_uniform_(self.d2_item.weight, gain=1)


        self.item_anchor_mode = str(
            args.params.get("item_anchor_mode", "free")
        ).strip().lower()
        if self.item_anchor_mode == "content_proj":
            if not self.use_tag_outputs_as_ui_input:
                raise ValueError(
                    "item_anchor_mode=content_proj is only meaningful when the "
                    "Tag graph feeds the U-I stage (use_tag_outputs_as_ui_input "
                    "= True). The wocross ablation bypasses the Tag path."
                )
            if not bool(args.params.get("use_pca_item_init", True)):
                raise ValueError(
                    "item_anchor_mode=content_proj requires use_pca_item_init=true."
                )
            self.d1_item_tg.weight.requires_grad_(False)
            self.d2_item_tg.weight.requires_grad_(False)
            self.d1_item_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            self.d2_item_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            nn.init.eye_(self.d1_item_proj.weight)
            nn.init.eye_(self.d2_item_proj.weight)
            print(
                "ABLAG: item_anchor_mode=content_proj — I-Tag anchor frozen to "
                "PCA content, learnable 128x128 projection added (identity-init)."
            )
        elif self.item_anchor_mode != "free":
            raise ValueError(
                f"item_anchor_mode must be 'free' or 'content_proj', got "
                f"{self.item_anchor_mode!r}."
            )


        self.shared_split_mode = str(
            args.params.get("shared_split_mode", "gate")
        ).strip().lower()
        if self.shared_split_mode not in ("gate", "tag2"):
            raise ValueError(
                f"shared_split_mode must be 'gate' or 'tag2', got "
                f"{self.shared_split_mode!r}."
            )
        if self.shared_split_mode == "tag2" and not self.use_tag_outputs_as_ui_input:

            print(
                "Info: shared_split_mode=tag2 has no Tag2 channel when the Tag "
                "path is bypassed; falling back to the gate split."
            )
            self.shared_split_mode = "gate"
        if self.shared_split_mode == "tag2":
            print(
                "ABLAG: shared_split_mode=tag2 — shared user branch is the Tag2 "
                "contribution, so shared_cross_bpr / shared MMD gradients flow "
                "through the shared Tag2 bridge."
            )


        self.tag_graph_topology = str(
            args.params.get("tag_graph_topology", "chain")
        ).strip().lower()
        if self.tag_graph_topology not in ("chain", "parallel", "swapped"):
            raise ValueError(
                "tag_graph_topology must be 'chain', 'parallel', or 'swapped', "
                f"got {self.tag_graph_topology!r}."
            )
        if self.tag_graph_topology == "parallel":
            print(
                "ABLAG: tag_graph_topology=parallel — Tag1 and Tag2 hang off the "
                "anchors as separate one-hop channels (Tag1-Tag2 edges dropped)."
            )
        elif self.tag_graph_topology == "swapped":
            print(
                "ABLAG: tag_graph_topology=swapped -- propagation uses "
                "Anchor-Tag2-Tag1 (Tag2 one hop, Tag1 two hops)."
            )


        nn.init.constant_(self.gate_d1[-1].bias, 0.0)
        nn.init.constant_(self.gate_d2[-1].bias, 0.0)
        nn.init.constant_(self.fuse_d1[-1].bias, 0.0)
        nn.init.constant_(self.fuse_d2[-1].bias, 0.0)


        self.has_tag_graphs = False
        self._build_tag_graphs(args)


        self._maybe_load_teacher_init(args)

        self.reg_weight = params.get("reg_weight", 0.001)
        self.lambda_u_tag_sparse = params.get("lambda_u_tag_sparse", 1e-5)
        self.lambda_i_tag_sparse = params.get("lambda_i_tag_sparse", 1e-5)
        self.align_weight = params.get("align_weight", 0.1)
        self.align_temp = params.get("align_temp", 0.2)
        self.ortho_weight = params.get("ortho_weight", 0.1)
        self.shared_domain_adv_weight = float(
            params.get("shared_domain_adv_weight", 0.0) or 0.0
        )
        self.private_domain_cls_weight = float(
            params.get("private_domain_cls_weight", 0.0) or 0.0
        )
        self.domain_grl_lambda = float(params.get("domain_grl_lambda", 1.0) or 1.0)


        self.lna_weight = params.get("lna_weight", 0.0)
        self.lna_temp = params.get("lna_temp", 0.2)


        self.pseudo_overlap_weight = params.get("pseudo_overlap_weight", 0.0)
        self.pseudo_overlap_refresh_every = int(
            params.get("pseudo_overlap_refresh_every", 3)
        )


        self.shared_user_nce_weight = params.get("shared_user_nce_weight", 0.0)


        self.shared_user_mmd_weight = float(
            params.get("shared_user_mmd_weight", 0.0) or 0.0
        )
        if self.shared_user_mmd_weight != 0:
            self.shared_user_mmd_weight_A = self.shared_user_mmd_weight
            self.shared_user_mmd_weight_B = self.shared_user_mmd_weight
        else:
            self.shared_user_mmd_weight_A = float(
                params.get("shared_user_mmd_weight_A", 0.0) or 0.0
            )
            self.shared_user_mmd_weight_B = float(
                params.get("shared_user_mmd_weight_B", 0.0) or 0.0
            )


        self.shared_overlap_cos_weight = params.get("shared_overlap_cos_weight", 0.0)


        self.shared_distill_weight = params.get("shared_distill_weight", 0.0)


        self.shared_bpr_weight = params.get("shared_bpr_weight", 0.0)

        self.shared_cross_bpr_weight = params.get("shared_cross_bpr_weight", 0.0)

        self.register_buffer(
            "_epoch", torch.zeros(1, dtype=torch.long), persistent=False,
        )
        self._last_pseudo_refresh = -1


        self.register_buffer(
            "_d1_partner_overlap", torch.zeros(0, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_d2_partner_overlap", torch.zeros(0, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_d1_partner_valid", torch.zeros(0, dtype=torch.bool), persistent=False,
        )
        self.register_buffer(
            "_d2_partner_valid", torch.zeros(0, dtype=torch.bool), persistent=False,
        )


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _make_domain_discriminator(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def _maybe_load_pca_item_init(self, args):
        if (
            not self.use_tag_outputs_as_ui_input
            or not bool(args.params.get("use_pca_item_init", True))
        ):
            return

        item_init = load_pca_item_initializers(
            args,
            self.emb_dim,
            dtype=self.dtype,
            validate_source=bool(
                args.params.get("pca_item_validate_source", False)
            ),
        )
        if item_init is None:
            raise FileNotFoundError(
                "PCA item initialisation is enabled, but neither a valid "
                f"cache nor source embeddings were found for {args.domains}."
            )

        d1_init = item_init["d1_item_init"].to(dtype=self.dtype)
        d2_init = item_init["d2_item_init"].to(dtype=self.dtype)
        d1_valid = item_init["d1_valid_mask"].bool()
        d2_valid = item_init["d2_valid_mask"].bool()
        with torch.no_grad():
            d1_rows = d1_valid.nonzero(as_tuple=False).squeeze(1)
            d2_rows = d2_valid.nonzero(as_tuple=False).squeeze(1)
            self.d1_item_tg.weight.index_copy_(
                0, d1_rows, d1_init.index_select(0, d1_rows),
            )
            self.d2_item_tg.weight.index_copy_(
                0, d2_rows, d2_init.index_select(0, d2_rows),
            )

        print(
            "Loaded PCA item embeddings into I-Tag tables: "
            f"d1={int(d1_valid.sum().item())}/{self.d1_n_items}, "
            f"d2={int(d2_valid.sum().item())}/{self.d2_n_items}, "
            f"source={item_init['source']}."
        )

    def _resize_codebook(self, x):
        x = x.to(dtype=self.dtype)
        if x.shape[1] < self.emb_dim:
            return F.pad(x, (0, self.emb_dim - x.shape[1]))
        if x.shape[1] > self.emb_dim:
            return x[:, :self.emb_dim]
        return x

    def _item_tg_weight(self, domain):

        if self.item_anchor_mode == "content_proj":
            proj = self.d1_item_proj if domain == 1 else self.d2_item_proj
            table = self.d1_item_tg if domain == 1 else self.d2_item_tg
            return proj(table.weight)
        return (self.d1_item_tg if domain == 1 else self.d2_item_tg).weight

    def _random_tag_embedding(self, n_tags):
        embedding = torch.empty(n_tags, self.emb_dim, dtype=self.dtype)
        nn.init.xavier_normal_(embedding)
        return embedding

    def _user_tag1_embedding(self, domain):
        if self.rqvae_codebook_sharing == "shared_shared":
            return self.u_tag1_emb
        return self.d1_u_tag1_emb if domain == 1 else self.d2_u_tag1_emb

    def _item_tag1_embedding(self, domain):
        if self.rqvae_codebook_sharing == "shared_shared":
            return self.i_tag1_emb
        return self.d1_i_tag1_emb if domain == 1 else self.d2_i_tag1_emb

    def _collect_user_tag1_edges(self, train_inters, codes, domain, code_col=0):

        shift = self.d1_n_users - self.n_shared_users
        pairs = {}
        for inter in train_inters:
            u, i = int(inter[0]), int(inter[1])
            if domain == 2 and u >= self.n_shared_users:
                u = u + shift
            key = (u, int(codes[i, code_col].item()))
            pairs[key] = pairs.get(key, 0) + 1
        if not pairs:
            return (torch.zeros((0, 2), dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32))
        edges = torch.tensor(list(pairs.keys()), dtype=torch.long)
        freqs = torch.tensor(list(pairs.values()), dtype=torch.float32)
        return edges, freqs

    @staticmethod
    def _item_interaction_counts(train_inters, n_items):
        counts = torch.zeros(n_items, dtype=torch.float32)
        for inter in train_inters:
            counts[int(inter[1])] += 1.0
        return counts

    @staticmethod
    def _max_normalize(x):
        return x / x.max().clamp(min=1.0)

    def _build_user_tag2_profile(self, train_inters, codes, domain, n_tag2,
                                 item_tag2_soft=None):

        shift = self.d1_n_users - self.n_shared_users
        profile = torch.zeros(self.n_users, n_tag2, dtype=torch.float32)
        if item_tag2_soft is not None:
            item_tag2_soft = item_tag2_soft.float()
            assert item_tag2_soft.shape[1] == n_tag2, (
                f"item_tag2_soft width {item_tag2_soft.shape[1]} != n_tag2 {n_tag2}"
            )

        for inter in train_inters:
            u, i = int(inter[0]), int(inter[1])
            if domain == 2 and u >= self.n_shared_users:
                u = u + shift
            if item_tag2_soft is None:
                profile[u, int(codes[i, 1].item())] += 1.0
            else:
                profile[u] += item_tag2_soft[i]

        return profile / profile.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _build_tag_graphs(self, args):

        rq_path = os.path.join(args.material_path, "rqvae_codes.pt")
        if not os.path.isfile(rq_path):
            print(
                f"Warning: rqvae codes not found at {rq_path}, "
                f"per-domain tag graphs disabled."
            )
            self.has_tag_graphs = False
            return

        rq = torch.load(rq_path, map_location="cpu")
        requested_sharing = str(
            args.params.get("rqvae_codebook_sharing", "private_shared")
        ).strip().lower()
        artifact_sharing = str(
            rq.get("codebook_sharing", "private_shared")
        ).strip().lower()
        if requested_sharing not in ("private_shared", "shared_shared"):
            raise ValueError(
                "rqvae_codebook_sharing must be 'private_shared' or "
                f"'shared_shared', got {requested_sharing!r}."
            )
        if artifact_sharing != requested_sharing:
            raise RuntimeError(
                f"RQ-VAE artifact uses {artifact_sharing!r}, but config requests "
                f"{requested_sharing!r}. Rerun rqvae.py before main.py."
            )
        self.rqvae_codebook_sharing = artifact_sharing
        d1_codes = rq["d1_item_rqcodes"].long()
        d2_codes = rq["d2_item_rqcodes"].long()
        d1_cb1 = rq["d1_codebook1"]
        d2_cb1 = rq["d2_codebook1"]
        shared_cb2 = rq["shared_codebook2"]
        d1_item_tag2_soft = rq.get("d1_item_tag2_soft")
        d2_item_tag2_soft = rq.get("d2_item_tag2_soft")


        tag_codes_mode = str(
            args.params.get("tag_codes", "none")
        ).strip().lower()
        if tag_codes_mode == "shuffle":
            g = torch.Generator().manual_seed(12345)
            d1_codes = d1_codes[torch.randperm(d1_codes.shape[0], generator=g)]
            d2_codes = d2_codes[torch.randperm(d2_codes.shape[0], generator=g)]
            print(
                f"ABLAG: tag_codes=shuffle — item->code assignments permuted "
                f"per domain (seed=12345). Cluster sizes and Tag1-Tag2 edges "
                f"preserved; item semantics broken."
            )
        elif tag_codes_mode != "none":
            raise ValueError(
                f"tag_codes must be 'none' or 'shuffle', got {tag_codes_mode!r}."
            )

        d1_n_tag1 = int(d1_cb1.shape[0])
        d2_n_tag1 = int(d2_cb1.shape[0])
        n_tag2 = int(shared_cb2.shape[0])
        if self.rqvae_codebook_sharing == "shared_shared":
            if d1_cb1.shape != d2_cb1.shape or not torch.equal(d1_cb1, d2_cb1):
                raise RuntimeError(
                    "S->S artifact must contain identical domain-1/domain-2 "
                    "Tag1 codebooks. Rerun rqvae.py."
                )


        tag_embedding_init = str(
            args.params.get("tag_embedding_init", "rqvae")
        ).strip().lower()
        if tag_embedding_init == "rqvae":
            d1_tag1_init = self._resize_codebook(d1_cb1)
            d2_tag1_init = self._resize_codebook(d2_cb1)
            tag2_init = self._resize_codebook(shared_cb2)
        elif tag_embedding_init == "random":
            d1_tag1_init = self._random_tag_embedding(d1_n_tag1)
            d2_tag1_init = (
                d1_tag1_init
                if self.rqvae_codebook_sharing == "shared_shared"
                else self._random_tag_embedding(d2_n_tag1)
            )
            tag2_init = self._random_tag_embedding(n_tag2)
        else:
            raise ValueError(
                "tag_embedding_init must be 'rqvae' or 'random', got "
                f"{tag_embedding_init!r}."
            )
        if self.rqvae_codebook_sharing == "shared_shared":
            self.u_tag1_emb = nn.Parameter(d1_tag1_init.clone())
            self.i_tag1_emb = nn.Parameter(d1_tag1_init.clone())
        else:
            self.d1_u_tag1_emb = nn.Parameter(d1_tag1_init.clone())
            self.d2_u_tag1_emb = nn.Parameter(d2_tag1_init.clone())
            self.d1_i_tag1_emb = nn.Parameter(d1_tag1_init.clone())
            self.d2_i_tag1_emb = nn.Parameter(d2_tag1_init.clone())
        self.u_tag2_emb = nn.Parameter(tag2_init.clone())
        self.i_tag2_emb = nn.Parameter(tag2_init.clone())

        d1_ut_edges, d1_ut_freqs = self._collect_user_tag1_edges(
            args.d1["train"], d1_codes, domain=1,
        )
        d2_ut_edges, d2_ut_freqs = self._collect_user_tag1_edges(
            args.d2["train"], d2_codes, domain=2,
        )
        self.register_buffer(
            "_d1_user_tag2_profile",
            self._build_user_tag2_profile(
                args.d1["train"], d1_codes, domain=1, n_tag2=n_tag2,
                item_tag2_soft=d1_item_tag2_soft,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_d2_user_tag2_profile",
            self._build_user_tag2_profile(
                args.d2["train"], d2_codes, domain=2, n_tag2=n_tag2,
                item_tag2_soft=d2_item_tag2_soft,
            ),
            persistent=False,
        )

        d1_it_edges = torch.stack(
            [torch.arange(self.d1_n_items, dtype=torch.long), d1_codes[:, 0]], dim=1,
        )
        d2_it_edges = torch.stack(
            [torch.arange(self.d2_n_items, dtype=torch.long), d2_codes[:, 0]], dim=1,
        )


        if self.tag_graph_topology in ("parallel", "swapped"):
            d1_ut2_edges, d1_ut2_freqs = self._collect_user_tag1_edges(
                args.d1["train"], d1_codes, domain=1, code_col=1,
            )
            d2_ut2_edges, d2_ut2_freqs = self._collect_user_tag1_edges(
                args.d2["train"], d2_codes, domain=2, code_col=1,
            )
            d1_it2_edges = torch.stack(
                [torch.arange(self.d1_n_items, dtype=torch.long), d1_codes[:, 1]],
                dim=1,
            )
            d2_it2_edges = torch.stack(
                [torch.arange(self.d2_n_items, dtype=torch.long), d2_codes[:, 1]],
                dim=1,
            )
        else:
            d1_ut2_edges = d2_ut2_edges = None
            d1_it2_edges = d2_it2_edges = None
            d1_ut2_freqs = d2_ut2_freqs = None

        d1_t1t2 = torch.stack([d1_codes[:, 0], d1_codes[:, 1]], dim=1)
        d2_t1t2 = torch.stack([d2_codes[:, 0], d2_codes[:, 1]], dim=1)

        d1_item_counts = self._item_interaction_counts(args.d1["train"], self.d1_n_items)
        d2_item_counts = self._item_interaction_counts(args.d2["train"], self.d2_n_items)
        d1_item_pw = self._max_normalize(d1_item_counts)
        d2_item_pw = self._max_normalize(d2_item_counts)

        d1_tag1_counts = torch.zeros(d1_n_tag1, dtype=torch.float32)
        d1_tag1_counts.scatter_add_(0, d1_codes[:, 0].long(), d1_item_counts)
        d2_tag1_counts = torch.zeros(d2_n_tag1, dtype=torch.float32)
        d2_tag1_counts.scatter_add_(0, d2_codes[:, 0].long(), d2_item_counts)
        d1_tag1_pw = self._max_normalize(d1_tag1_counts)
        d2_tag1_pw = self._max_normalize(d2_tag1_counts)

        d1_it_sparse_w = d1_item_pw[d1_it_edges[:, 0].long()]
        d2_it_sparse_w = d2_item_pw[d2_it_edges[:, 0].long()]
        d1_ut_sparse_w = (
            d1_tag1_pw[d1_ut_edges[:, 1].long()]
            if d1_ut_edges.numel() else torch.zeros(0, dtype=torch.float32)
        )
        d2_ut_sparse_w = (
            d2_tag1_pw[d2_ut_edges[:, 1].long()]
            if d2_ut_edges.numel() else torch.zeros(0, dtype=torch.float32)
        )

        if self.tag_graph_topology in ("parallel", "swapped"):
            d1_tag2_counts = torch.zeros(n_tag2, dtype=torch.float32)
            d1_tag2_counts.scatter_add_(0, d1_codes[:, 1].long(), d1_item_counts)
            d2_tag2_counts = torch.zeros(n_tag2, dtype=torch.float32)
            d2_tag2_counts.scatter_add_(0, d2_codes[:, 1].long(), d2_item_counts)
            d1_tag2_pw = self._max_normalize(d1_tag2_counts)
            d2_tag2_pw = self._max_normalize(d2_tag2_counts)

            d1_it2_sparse_w = d1_item_pw[d1_it2_edges[:, 0].long()]
            d2_it2_sparse_w = d2_item_pw[d2_it2_edges[:, 0].long()]
            d1_ut2_sparse_w = (
                d1_tag2_pw[d1_ut2_edges[:, 1].long()]
                if d1_ut2_edges.numel() else torch.zeros(0, dtype=torch.float32)
            )
            d2_ut2_sparse_w = (
                d2_tag2_pw[d2_ut2_edges[:, 1].long()]
                if d2_ut2_edges.numel() else torch.zeros(0, dtype=torch.float32)
            )
        else:
            d1_it2_sparse_w = d2_it2_sparse_w = None
            d1_ut2_sparse_w = d2_ut2_sparse_w = None

        topo = self.tag_graph_topology
        self.d1_u_tag_graph = TagGraph(
            n_anchors=self.n_users, n_tag1=d1_n_tag1, n_tag2=n_tag2,
            at1_edges=d1_ut_edges, at1_freqs=d1_ut_freqs,
            t1t2_edges=d1_t1t2, dtype=self.dtype, freq_init=True,
            at1_sparsity_weights=d1_ut_sparse_w,
            at2_edges=d1_ut2_edges, at2_freqs=d1_ut2_freqs,
            at2_sparsity_weights=d1_ut2_sparse_w, topology=topo,
        )
        self.d2_u_tag_graph = TagGraph(
            n_anchors=self.n_users, n_tag1=d2_n_tag1, n_tag2=n_tag2,
            at1_edges=d2_ut_edges, at1_freqs=d2_ut_freqs,
            t1t2_edges=d2_t1t2, dtype=self.dtype, freq_init=True,
            at1_sparsity_weights=d2_ut_sparse_w,
            at2_edges=d2_ut2_edges, at2_freqs=d2_ut2_freqs,
            at2_sparsity_weights=d2_ut2_sparse_w, topology=topo,
        )
        self.d1_i_tag_graph = TagGraph(
            n_anchors=self.d1_n_items, n_tag1=d1_n_tag1, n_tag2=n_tag2,
            at1_edges=d1_it_edges, at1_freqs=None,
            t1t2_edges=d1_t1t2, dtype=self.dtype, freq_init=False,
            at1_sparsity_weights=d1_it_sparse_w,
            at2_edges=d1_it2_edges, at2_freqs=None,
            at2_sparsity_weights=d1_it2_sparse_w, topology=topo,
        )
        self.d2_i_tag_graph = TagGraph(
            n_anchors=self.d2_n_items, n_tag1=d2_n_tag1, n_tag2=n_tag2,
            at1_edges=d2_it_edges, at1_freqs=None,
            t1t2_edges=d2_t1t2, dtype=self.dtype, freq_init=False,
            at1_sparsity_weights=d2_it_sparse_w,
            at2_edges=d2_it2_edges, at2_freqs=None,
            at2_sparsity_weights=d2_it2_sparse_w, topology=topo,
        )

        self.has_tag_graphs = True
        at2_info = (
            f", d1_ut2={d1_ut2_edges.shape[0]}, d2_ut2={d2_ut2_edges.shape[0]}"
            if topo in ("parallel", "swapped") else ""
        )
        print(
            f"Loaded rqvae codes from {rq_path}: "
            f"codebook_sharing={self.rqvae_codebook_sharing}, "
            f"tag_embedding_init={tag_embedding_init}, topology={topo}, "
            f"d1_n_tag1={d1_n_tag1}, d2_n_tag1={d2_n_tag1}, n_tag2={n_tag2}, "
            f"d1_ut={d1_ut_edges.shape[0]}, d2_ut={d2_ut_edges.shape[0]}, "
            f"d1_it={d1_it_edges.shape[0]}, d2_it={d2_it_edges.shape[0]}"
            f"{at2_info}."
        )

    def _maybe_load_teacher_init(self, args):

        if not args.params.get("use_teacher_pretrain", False):
            print("Info: teacher_pretrain transfer disabled; "
                  "keeping non-teacher initialisation.")
            return

        path = os.path.join(args.material_path, "teacher_pretrain.pt")
        if not os.path.isfile(path):
            print(f"Info: teacher_pretrain.pt not found at {path}; "
                  f"keeping non-teacher initialisation.")
            return
        ckpt = torch.load(path, map_location="cpu")
        if not self.has_tag_graphs:
            print(f"Info: tag graphs disabled, skipping teacher transfer "
                  f"from {path}.")
            return
        loaded = []
        with torch.no_grad():

            item_pairs = []
            if self.item_anchor_mode != "content_proj":
                item_pairs = [
                    (("d1_item_init",), self.d1_item_tg),
                    (("d2_item_init",), self.d2_item_tg),
                ]
            for keys, emb in [
                (("d1_user_init", "mix_user_init"), self.d1_user_tg),
                (("d2_user_init", "mix_user_init"), self.d2_user_tg),
            ] + item_pairs:
                for key in keys:
                    t = ckpt.get(key)
                    if t is not None and tuple(t.shape) == tuple(emb.weight.shape):
                        emb.weight.data.copy_(t.to(dtype=self.dtype))
                        loaded.append(key)
                        break

            for key, graph in [
                ("d1_ut_logits", self.d1_u_tag_graph),
                ("d2_ut_logits", self.d2_u_tag_graph),
                ("d1_it_logits", self.d1_i_tag_graph),
                ("d2_it_logits", self.d2_i_tag_graph),
            ]:
                t = ckpt.get(key)
                if t is not None and t.numel() == graph.at1_logits.numel():
                    graph.at1_logits.data.copy_(t.to(dtype=self.dtype))
                    loaded.append(key)

        if loaded:
            print(f"Loaded teacher Tag-branch init from {path}: "
                  f"{', '.join(loaded)}. Role-specific Tag embeddings kept "
                  f"at their own init.")
        else:
            print(f"Warning: no compatible teacher Tag-branch init found in {path}; "
                  f"keeping non-teacher initialisation.")


    def _build_norm_graph(self, sp_mat):
        norm = symmetric_normalize(sp_mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _build_overlap_pos_lookup(self, args):

        def _build(train, n_shared, n_items):
            offsets = torch.zeros(n_shared + 1, dtype=torch.long)
            if n_shared == 0:
                return (
                    offsets,
                    torch.empty(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.bool),
                    torch.empty(0, dtype=torch.long),
                )
            buckets = [set() for _ in range(n_shared)]
            for row in train:
                u = int(row[0])
                if 0 <= u < n_shared:
                    buckets[u].add(int(row[1]))
            counts = torch.tensor([len(b) for b in buckets], dtype=torch.long)
            offsets[1:] = torch.cumsum(counts, dim=0)
            flat = torch.empty(int(counts.sum().item()), dtype=torch.long)
            for u, b in enumerate(buckets):
                if b:
                    flat[offsets[u]:offsets[u + 1]] = torch.tensor(sorted(b), dtype=torch.long)
            has_pos = counts > 0
            user_ids = torch.arange(n_shared, dtype=torch.long).repeat_interleave(counts)
            pos_keys = user_ids * int(n_items) + flat
            return offsets, flat, has_pos, pos_keys

        d1_off, d1_items, d1_has, d1_keys = _build(
            args.d1["train"], self.n_shared_users, self.d1_n_items,
        )
        d2_off, d2_items, d2_has, d2_keys = _build(
            args.d2["train"], self.n_shared_users, self.d2_n_items,
        )

        self.register_buffer("_d1_ov_pos_offsets", d1_off, persistent=False)
        self.register_buffer("_d1_ov_pos_items", d1_items, persistent=False)
        self.register_buffer("_d1_ov_has_pos", d1_has, persistent=False)
        self.register_buffer("_d1_ov_pos_keys", d1_keys, persistent=False)
        self.register_buffer("_d2_ov_pos_offsets", d2_off, persistent=False)
        self.register_buffer("_d2_ov_pos_items", d2_items, persistent=False)
        self.register_buffer("_d2_ov_has_pos", d2_has, persistent=False)
        self.register_buffer("_d2_ov_pos_keys", d2_keys, persistent=False)

    def _build_all_pos_lookup(self, args):

        def _build(train, n_users_local):
            offsets = torch.zeros(n_users_local + 1, dtype=torch.long)
            if n_users_local == 0:
                return offsets, torch.empty(0, dtype=torch.long), torch.zeros(0, dtype=torch.bool)
            buckets = [set() for _ in range(n_users_local)]
            for row in train:
                u = int(row[0])
                if 0 <= u < n_users_local:
                    buckets[u].add(int(row[1]))
            counts = torch.tensor([len(b) for b in buckets], dtype=torch.long)
            offsets[1:] = torch.cumsum(counts, dim=0)
            flat = torch.empty(int(counts.sum().item()), dtype=torch.long)
            for u, b in enumerate(buckets):
                if b:
                    flat[offsets[u]:offsets[u + 1]] = torch.tensor(sorted(b), dtype=torch.long)
            has_pos = counts > 0
            return offsets, flat, has_pos

        d1_off, d1_items, d1_has = _build(args.d1["train"], self.d1_n_users)
        d2_off, d2_items, d2_has = _build(args.d2["train"], self.d2_n_users)

        self.register_buffer("_d1_all_pos_offsets", d1_off, persistent=False)
        self.register_buffer("_d1_all_pos_items", d1_items, persistent=False)
        self.register_buffer("_d1_all_has_pos", d1_has, persistent=False)
        self.register_buffer("_d2_all_pos_offsets", d2_off, persistent=False)
        self.register_buffer("_d2_all_pos_items", d2_items, persistent=False)
        self.register_buffer("_d2_all_has_pos", d2_has, persistent=False)

    def _gather_pos_items_all(self, users, domain):

        if domain == 1:
            offsets, flat, has_pos = (
                self._d1_all_pos_offsets, self._d1_all_pos_items, self._d1_all_has_pos,
            )
        else:
            offsets, flat, has_pos = (
                self._d2_all_pos_offsets, self._d2_all_pos_items, self._d2_all_has_pos,
            )

        valid_anchor = has_pos[users]
        starts = offsets[users]
        counts = offsets[users + 1] - starts
        if counts.numel() == 0:
            return (
                torch.empty(0, 0, dtype=torch.long, device=users.device),
                torch.empty(0, 0, dtype=torch.bool, device=users.device),
                valid_anchor,
            )
        max_count = int(counts.max().item())
        if max_count == 0:
            return (
                torch.empty(users.shape[0], 0, dtype=torch.long, device=users.device),
                torch.empty(users.shape[0], 0, dtype=torch.bool, device=users.device),
                valid_anchor,
            )

        ar = torch.arange(max_count, device=users.device)
        mask = ar.unsqueeze(0) < counts.unsqueeze(1)
        flat_idx = starts.unsqueeze(1) + ar.unsqueeze(0)
        flat_idx = torch.where(mask, flat_idx, torch.zeros_like(flat_idx))
        items = flat[flat_idx]
        return items, mask, valid_anchor


    def aggregate(self, graph, all_emb, n_layers):
        embeddings = [all_emb]
        for _ in range(n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            all_emb = self.dropout(all_emb)
            embeddings.append(all_emb)
        return torch.stack(embeddings, dim=1).mean(dim=1)


    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[:self.d1_n_users], emb[self.n_users:]

        user = torch.cat([emb[:self.n_shared_users], emb[self.d1_n_users:self.n_users]], dim=0)
        return user, emb[self.n_users:]


    def forward(self):

        split_on_tag2 = (
            self.shared_split_mode == "tag2" and self.has_tag_graphs
        )
        d1_user_t2 = d2_user_t2 = None
        d1_item_t2 = d2_item_t2 = None

        if self.has_tag_graphs and split_on_tag2:

            d1_u_base, d1_u_t1, d1_user_t2 = self.d1_u_tag_graph.propagate_decomposed(
                self.d1_user_tg.weight, self._user_tag1_embedding(1), self.u_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d2_u_base, d2_u_t1, d2_user_t2 = self.d2_u_tag_graph.propagate_decomposed(
                self.d2_user_tg.weight, self._user_tag1_embedding(2), self.u_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d1_i_base, d1_i_t1, d1_item_t2 = self.d1_i_tag_graph.propagate_decomposed(
                self._item_tg_weight(1), self._item_tag1_embedding(1), self.i_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d2_i_base, d2_i_t1, d2_item_t2 = self.d2_i_tag_graph.propagate_decomposed(
                self._item_tg_weight(2), self._item_tag1_embedding(2), self.i_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d1_user_tag = d1_u_base + d1_u_t1 + d1_user_t2
            d2_user_tag = d2_u_base + d2_u_t1 + d2_user_t2
            d1_item_tag = d1_i_base + d1_i_t1 + d1_item_t2
            d2_item_tag = d2_i_base + d2_i_t1 + d2_item_t2
        elif self.has_tag_graphs:
            d1_user_tag = self.d1_u_tag_graph.propagate(
                self.d1_user_tg.weight, self._user_tag1_embedding(1), self.u_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d2_user_tag = self.d2_u_tag_graph.propagate(
                self.d2_user_tg.weight, self._user_tag1_embedding(2), self.u_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d1_item_tag = self.d1_i_tag_graph.propagate(
                self._item_tg_weight(1), self._item_tag1_embedding(1), self.i_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
            d2_item_tag = self.d2_i_tag_graph.propagate(
                self._item_tg_weight(2), self._item_tag1_embedding(2), self.i_tag2_emb,
                self.intra_domain_layers, self.dropout,
            )
        else:
            d1_user_tag = self.d1_user.weight
            d2_user_tag = self.d2_user.weight
            d1_item_tag = self.d1_item.weight
            d2_item_tag = self.d2_item.weight


        d1_all = torch.cat([d1_user_tag, d1_item_tag])
        d2_all = torch.cat([d2_user_tag, d2_item_tag])
        if split_on_tag2:
            d1_all = torch.cat(
                [d1_all, torch.cat([d1_user_t2, d1_item_t2])], dim=1,
            )
            d2_all = torch.cat(
                [d2_all, torch.cat([d2_user_t2, d2_item_t2])], dim=1,
            )
        d1_emb_lgcn = self.aggregate(self.d1_graph, d1_all, self.ui_graph_layers)
        d2_emb_lgcn = self.aggregate(self.d2_graph, d2_all, self.ui_graph_layers)
        if split_on_tag2:
            d1_emb_t2 = d1_emb_lgcn[:, self.emb_dim:]
            d2_emb_t2 = d2_emb_lgcn[:, self.emb_dim:]
            d1_emb_lgcn = d1_emb_lgcn[:, :self.emb_dim]
            d2_emb_lgcn = d2_emb_lgcn[:, :self.emb_dim]

        d1_user_raw, d1_item = self._split_domain_ui(d1_emb_lgcn, 1)
        d2_user_raw, d2_item = self._split_domain_ui(d2_emb_lgcn, 2)

        if split_on_tag2:

            s1, _ = self._split_domain_ui(d1_emb_t2, 1)
            s2, _ = self._split_domain_ui(d2_emb_t2, 2)
            p1 = d1_user_raw - s1
            p2 = d2_user_raw - s2
        else:
            g1 = torch.sigmoid(self.gate_d1(d1_user_raw))
            g2 = torch.sigmoid(self.gate_d2(d2_user_raw))
            p1 = g1 * d1_user_raw
            s1 = (1 - g1) * d1_user_raw
            p2 = g2 * d2_user_raw
            s2 = (1 - g2) * d2_user_raw

        alpha1 = torch.sigmoid(self.fuse_d1(torch.cat([p1, s1], dim=-1)))
        d1_user = alpha1 * s1 + (1 - alpha1) * p1
        alpha2 = torch.sigmoid(self.fuse_d2(torch.cat([p2, s2], dim=-1)))
        d2_user = alpha2 * s2 + (1 - alpha2) * p2

        return d1_user, d1_item, d2_user, d2_item, s1, s2, p1, p2

    def _xd_infonce_loss(self, s1, s2, inter_d1, inter_d2,
                         d1_item_emb, d2_item_emb):

        if self.n_shared_users == 0:
            return torch.tensor(0.0, device=self.device)

        d1_n = int(inter_d1.shape[0])
        d2_n = int(inter_d2.shape[0])
        if d1_n + d2_n == 0:
            return torch.tensor(0.0, device=self.device)

        d1_batch_pos = (
            d1_item_emb[inter_d1[:, 1]]
            if d1_n
            else d1_item_emb.new_empty((0, d1_item_emb.shape[1]))
        )
        d2_batch_pos = (
            d2_item_emb[inter_d2[:, 1]]
            if d2_n
            else d2_item_emb.new_empty((0, d2_item_emb.shape[1]))
        )
        batch_pos_pool = torch.cat([d1_batch_pos, d2_batch_pos], dim=0)

        losses = [
            self._xd_infonce_one_side(
                inter_self=inter_d1,
                own_shared=s1,
                self_item_emb=d1_item_emb,
                other_item_emb=d2_item_emb,
                other_domain=2,
                batch_pos_pool=batch_pos_pool,
                self_pool_offset=0,
                other_pool_offset=d1_n,
                other_batch_items=inter_d2[:, 1] if d2_n else None,
            ),
            self._xd_infonce_one_side(
                inter_self=inter_d2,
                own_shared=s2,
                self_item_emb=d2_item_emb,
                other_item_emb=d1_item_emb,
                other_domain=1,
                batch_pos_pool=batch_pos_pool,
                self_pool_offset=d1_n,
                other_pool_offset=0,
                other_batch_items=inter_d1[:, 1] if d1_n else None,
            ),
        ]
        valid = [l for l in losses if l is not None]
        if not valid:
            return torch.tensor(0.0, device=self.device)
        return sum(valid) / len(valid)

    def _xd_infonce_one_side(self, inter_self, own_shared,
                             self_item_emb, other_item_emb, other_domain,
                             batch_pos_pool, self_pool_offset,
                             other_pool_offset, other_batch_items):
        if inter_self.numel() == 0:
            return None

        users_self = inter_self[:, 0]
        items_self = inter_self[:, 1]
        ov_mask = users_self < self.n_shared_users
        if not ov_mask.any():
            return None

        row_idx = torch.nonzero(ov_mask, as_tuple=False).squeeze(1)
        anchor_users = users_self[ov_mask]
        self_pos_items = items_self[ov_mask]


        other_pos_items, has_pos = self._sample_one_overlap_pos(
            anchor_users, other_domain,
        )
        if not has_pos.any():
            return None
        anchor_users = anchor_users[has_pos]
        self_pos_items = self_pos_items[has_pos]
        other_pos_items = other_pos_items[has_pos]
        row_idx = row_idx[has_pos]

        u_e = F.normalize(own_shared[anchor_users], p=2, dim=1)
        self_e = F.normalize(self_item_emb[self_pos_items], p=2, dim=1)
        other_e = F.normalize(other_item_emb[other_pos_items], p=2, dim=1)
        neg_e = F.normalize(batch_pos_pool, p=2, dim=1)

        self_mask = torch.zeros(
            u_e.shape[0], batch_pos_pool.shape[0],
            dtype=torch.bool, device=u_e.device,
        )
        self_cols = self_pool_offset + row_idx
        self_mask[torch.arange(u_e.shape[0], device=u_e.device), self_cols] = True

        losses = [
            self._one_positive_infonce(u_e, self_e, neg_e, self_mask),
        ]

        if other_batch_items is not None and other_batch_items.numel() > 0:
            other_mask_local = other_pos_items.unsqueeze(1) == other_batch_items.unsqueeze(0)
            other_mask = torch.zeros(
                u_e.shape[0], batch_pos_pool.shape[0],
                dtype=torch.bool, device=u_e.device,
            )
            end = other_pool_offset + other_batch_items.shape[0]
            other_mask[:, other_pool_offset:end] = other_mask_local
        else:
            other_mask = None
        losses.append(self._one_positive_infonce(u_e, other_e, neg_e, other_mask))
        return sum(losses) / len(losses)

    def _one_positive_infonce(self, anchor_e, pos_e, neg_e, neg_mask=None):
        pos_logits = (anchor_e * pos_e).sum(dim=1, keepdim=True) / self.align_temp
        neg_logits = (anchor_e @ neg_e.t()) / self.align_temp
        if neg_mask is not None:
            neg_logits = neg_logits.masked_fill(neg_mask, float("-inf"))
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(anchor_e.shape[0], dtype=torch.long, device=anchor_e.device)
        return F.cross_entropy(logits, labels)

    def _orthogonal_loss(self, p, s):
        p_norm = F.normalize(p, dim=-1)
        s_norm = F.normalize(s, dim=-1)
        return (p_norm * s_norm).sum(dim=-1).pow(2).mean()

    def _domain_classification_loss(self, z1, z2, inter_d1, inter_d2,
                                    discriminator, grl_lambda=None):
        feats = []
        labels = []
        if inter_d1.numel() > 0:
            u1 = torch.unique(inter_d1[:, 0])
            feats.append(z1[u1])
            labels.append(torch.zeros(u1.shape[0], dtype=torch.long, device=z1.device))
        if inter_d2.numel() > 0:
            u2 = torch.unique(inter_d2[:, 0])
            feats.append(z2[u2])
            labels.append(torch.ones(u2.shape[0], dtype=torch.long, device=z2.device))
        if not feats:
            return torch.tensor(0.0, device=self.device)

        x = torch.cat(feats, dim=0)
        y = torch.cat(labels, dim=0)
        if grl_lambda is not None:
            x = _grad_reverse(x, grl_lambda)
        return F.cross_entropy(discriminator(x), y)

    def _shared_user_nce_loss(self, s1, s2, inter_d1, inter_d2):

        if self.n_shared_users <= 0:
            return torch.tensor(0.0, device=self.device)
        if not (inter_d1.numel() or inter_d2.numel()):
            return torch.tensor(0.0, device=self.device)

        losses = []
        l1 = self._shared_user_nce_one_side(inter_d1[:, 0], inter_d2[:, 0], s1, s2)
        if l1 is not None:
            losses.append(l1)
        l2 = self._shared_user_nce_one_side(inter_d2[:, 0], inter_d1[:, 0], s2, s1)
        if l2 is not None:
            losses.append(l2)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return sum(losses) / len(losses)

    def _shared_user_nce_one_side(self, batch_users, other_batch_users, s_self, s_other):

        unique_users = torch.unique(batch_users)
        if unique_users.numel() == 0:
            return None

        overlap_mask = unique_users < self.n_shared_users
        if not overlap_mask.any():
            return None
        anchors = unique_users[overlap_mask]

        a = F.normalize(s_self[anchors], p=2, dim=1)
        overlap_ids = torch.arange(self.n_shared_users, device=batch_users.device)
        candidate_ids = torch.unique(
            torch.cat([overlap_ids, other_batch_users]), sorted=True,
        )
        b = F.normalize(s_other[candidate_ids], p=2, dim=1)
        logits = (a @ b.t()) / self.align_temp

        labels = torch.searchsorted(candidate_ids, anchors)
        return F.cross_entropy(logits, labels)

    def _shared_user_mmd_loss_terms(self, s1, s2, inter_d1, inter_d2):

        if not (inter_d1.numel() and inter_d2.numel()):
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        u1 = torch.unique(inter_d1[:, 0])
        u2 = torch.unique(inter_d2[:, 0])


        MIN_USERS = 4
        if u1.numel() < MIN_USERS or u2.numel() < MIN_USERS:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        x = F.normalize(s1[u1], p=2, dim=1)
        y = F.normalize(s2[u2], p=2, dim=1)
        return (
            self._mmd_rbf_multi(x, y.detach()),
            self._mmd_rbf_multi(x.detach(), y),
        )

    def _overlap_cos_loss(self, s1, s2, inter_d1, inter_d2):
        if self.n_shared_users <= 0:
            return torch.tensor(0.0, device=self.device)

        u1 = inter_d1[:, 0]
        u2 = inter_d2[:, 0]
        overlap_users = torch.unique(torch.cat([
            u1[u1 < self.n_shared_users],
            u2[u2 < self.n_shared_users],
        ]))
        if overlap_users.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        z1 = F.normalize(s1[overlap_users], p=2, dim=1)
        z2 = F.normalize(s2[overlap_users], p=2, dim=1)
        return 1.0 - (z1 * z2).sum(dim=1).mean()

    @staticmethod
    def _mmd_rbf_multi(x, y, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):

        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)
        loss = x.new_zeros(())
        for s in sigmas:
            gamma = 1.0 / (2.0 * s * s)
            loss = loss + (
                torch.exp(-gamma * xx).mean()
                + torch.exp(-gamma * yy).mean()
                - 2.0 * torch.exp(-gamma * xy).mean()
            )
        return loss / len(sigmas)

    def _lna_private_domain(self, users, items, private_emb, item_emb):

        if users.numel() < 2:
            return torch.tensor(0.0, device=self.device)

        u_e = F.normalize(private_emb[users], p=2, dim=1)
        i_e = F.normalize(item_emb[items], p=2, dim=1)
        logits = (u_e @ i_e.t()) / self.lna_temp

        labels = torch.arange(logits.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def get_score(self, user_emb, item_emb):
        return torch.mul(user_emb, item_emb).sum(dim=-1)

    def get_bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_scores = self.get_score(user_emb, pos_item_emb)
        neg_scores = self.get_score(user_emb, neg_item_emb)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _shared_rank_distill_loss(self, fused_user, shared_user,
                                  pos_item_emb, neg_item_emb):
        teacher_margin = (
            self.get_score(fused_user.detach(), pos_item_emb.detach())
            - self.get_score(fused_user.detach(), neg_item_emb.detach())
        )
        student_margin = (
            self.get_score(shared_user, pos_item_emb.detach())
            - self.get_score(shared_user, neg_item_emb.detach())
        )
        return F.mse_loss(student_margin, teacher_margin)

    def _shared_cross_bpr_one_side(self, batch_users, own_shared,
                                   other_item_emb, other_domain):
        if self.n_shared_users <= 0 or batch_users.numel() == 0:
            return None

        overlap_users = torch.unique(batch_users[batch_users < self.n_shared_users])
        if overlap_users.numel() == 0:
            return None

        users, pos_items, neg_items = self._sample_overlap_bpr_items(
            overlap_users, other_domain,
        )
        if users.numel() == 0:
            return None

        return self.get_bpr_loss(
            own_shared[users],
            other_item_emb[pos_items],
            other_item_emb[neg_items],
        )

    def _shared_cross_bpr_loss_terms(self, s1, s2, inter_d1, neg_d1_item,
                                     inter_d2, neg_d2_item, d1_item_emb,
                                     d2_item_emb):
        d1_terms = []
        d2_terms = []

        if inter_d1.numel() > 0:
            d1_terms.append(self.get_bpr_loss(
                s1[inter_d1[:, 0]],
                d1_item_emb[inter_d1[:, 1]],
                d1_item_emb[neg_d1_item],
            ))
        if inter_d2.numel() > 0:
            d2_terms.append(self.get_bpr_loss(
                s2[inter_d2[:, 0]],
                d2_item_emb[inter_d2[:, 1]],
                d2_item_emb[neg_d2_item],
            ))

        l1 = self._shared_cross_bpr_one_side(
            inter_d1[:, 0], s1, d2_item_emb, other_domain=2,
        )
        if l1 is not None:
            d1_terms.append(l1)
        l2 = self._shared_cross_bpr_one_side(
            inter_d2[:, 0], s2, d1_item_emb, other_domain=1,
        )
        if l2 is not None:
            d2_terms.append(l2)

        return d1_terms, d2_terms

    def get_bpr_reg_loss(self, users, pos_items, neg_items, domain):
        use_tag_base = self.has_tag_graphs and self.use_tag_outputs_as_ui_input
        user_ids = users
        if domain == 1:
            user_table = self.d1_user_tg if use_tag_base else self.d1_user
            item_anchor = self._item_tg_weight(1) if use_tag_base else self.d1_item.weight
        else:
            shift = self.d1_n_users - self.n_shared_users
            user_ids = torch.where(users < self.n_shared_users, users, users + shift)
            user_table = self.d2_user_tg if use_tag_base else self.d2_user
            item_anchor = self._item_tg_weight(2) if use_tag_base else self.d2_item.weight

        user_emb = user_table.weight[user_ids]
        pos_item_emb = item_anchor[pos_items]
        neg_item_emb = item_anchor[neg_items]

        return (
            user_emb.norm(2).pow(2)
            + pos_item_emb.norm(2).pow(2)
            + neg_item_emb.norm(2).pow(2)
        ) / (2 * len(users))

    @torch.no_grad()
    def _maybe_refresh_pseudo_pairs(self):

        if not self.training or not self.has_tag_graphs:
            return

        if self.pseudo_overlap_weight <= 0:
            return

        cur = int(self._epoch[0].item())
        if cur == self._last_pseudo_refresh:
            return
        if cur % self.pseudo_overlap_refresh_every != 0:
            return

        device = self.device
        n_s = self.n_shared_users
        self._last_pseudo_refresh = cur

        if n_s == 0:
            self._d1_partner_overlap = torch.zeros(self.d1_n_users, dtype=torch.long, device=device)
            self._d2_partner_overlap = torch.zeros(self.d2_n_users, dtype=torch.long, device=device)
            self._d1_partner_valid = torch.zeros(self.d1_n_users, dtype=torch.bool, device=device)
            self._d2_partner_valid = torch.zeros(self.d2_n_users, dtype=torch.bool, device=device)
            print(f"\n[Pseudo-overlap] Epoch {cur}: n_shared=0, partner pool empty -> disabled.")
            return

        profile_d1 = self._d1_user_tag2_profile.float()
        profile_d2 = self._d2_user_tag2_profile.float()

        p1 = profile_d1[:self.d1_n_users]
        d2_global_ids = torch.cat([
            torch.arange(n_s, device=device),
            torch.arange(self.d1_n_users, self.n_users, device=device),
        ])
        p2 = profile_d2[d2_global_ids]

        p1_ov = profile_d1[:n_s]
        p2_ov = profile_d2[:n_s]

        z1 = F.normalize(p1, p=2, dim=-1)
        z2 = F.normalize(p2, p=2, dim=-1)
        z1_ov = F.normalize(p1_ov, p=2, dim=-1)
        z2_ov = F.normalize(p2_ov, p=2, dim=-1)

        sim_d1 = z1 @ z2_ov.t()
        d1_partner = sim_d1.argmax(dim=1).long()
        d1_best = sim_d1.gather(1, d1_partner.unsqueeze(1)).squeeze(1)

        sim_d2 = z2 @ z1_ov.t()
        d2_partner = sim_d2.argmax(dim=1).long()
        d2_best = sim_d2.gather(1, d2_partner.unsqueeze(1)).squeeze(1)


        d1_valid = self._d2_ov_has_pos[d1_partner].clone()
        d2_valid = self._d1_ov_has_pos[d2_partner].clone()

        self._d1_partner_overlap = d1_partner
        self._d2_partner_overlap = d2_partner
        self._d1_partner_valid = d1_valid
        self._d2_partner_valid = d2_valid


        d1_self_match = ((d1_partner == torch.arange(self.d1_n_users, device=device)) & (torch.arange(self.d1_n_users, device=device) < n_s)).sum().item()
        d2_self_match = ((d2_partner == torch.arange(self.d2_n_users, device=device)) & (torch.arange(self.d2_n_users, device=device) < n_s)).sum().item()
        qs = torch.tensor([0.50, 0.90, 0.95, 0.99], device=device)
        d1_pcts = torch.quantile(d1_best.float(), qs).tolist()
        d2_pcts = torch.quantile(d2_best.float(), qs).tolist()
        print(
            f"\n[Pseudo-overlap] Epoch {cur}: partner pool = {n_s} overlap users; "
            f"d1_total={self.d1_n_users} (self_match={d1_self_match}, "
            f"valid={int(d1_valid.sum().item())}); "
            f"d2_total={self.d2_n_users} (self_match={d2_self_match}, "
            f"valid={int(d2_valid.sum().item())})."
        )
        print(
            f"  [diag] d1-anchor best-sim: max={d1_best.max().item():.3f}, "
            f"p50/p90/p95/p99={d1_pcts[0]:.3f}/{d1_pcts[1]:.3f}/{d1_pcts[2]:.3f}/{d1_pcts[3]:.3f}; "
            f"d2-anchor best-sim: max={d2_best.max().item():.3f}, "
            f"p50/p90/p95/p99={d2_pcts[0]:.3f}/{d2_pcts[1]:.3f}/{d2_pcts[2]:.3f}/{d2_pcts[3]:.3f}."
        )

    def _sample_one_overlap_pos(self, overlap_users, domain):

        if domain == 1:
            offsets = self._d1_ov_pos_offsets
            items = self._d1_ov_pos_items
            has_pos = self._d1_ov_has_pos
        else:
            offsets = self._d2_ov_pos_offsets
            items = self._d2_ov_pos_items
            has_pos = self._d2_ov_has_pos

        valid = has_pos[overlap_users]
        starts = offsets[overlap_users]
        counts = (offsets[overlap_users + 1] - starts).clamp(min=1)
        rand_offset = (
            torch.rand(overlap_users.shape[0], device=overlap_users.device)
            * counts.to(torch.float)
        ).long().clamp(max=counts - 1)
        pos_items = items[starts + rand_offset]
        return pos_items, valid

    def _overlap_pos_membership(self, overlap_users, items, domain):
        if domain == 1:
            pos_keys = self._d1_ov_pos_keys
            n_items = self.d1_n_items
        else:
            pos_keys = self._d2_ov_pos_keys
            n_items = self.d2_n_items

        if pos_keys.numel() == 0 or overlap_users.numel() == 0:
            return torch.zeros_like(items, dtype=torch.bool)

        keys = overlap_users.long() * int(n_items) + items.long()
        idx = torch.searchsorted(pos_keys, keys)
        in_bounds = idx < pos_keys.numel()
        idx = idx.clamp(max=pos_keys.numel() - 1)
        return in_bounds & (pos_keys[idx] == keys)

    def _sample_overlap_bpr_items(self, overlap_users, domain):

        if overlap_users.numel() == 0:
            empty = overlap_users.new_empty(0)
            return empty, empty, empty

        if domain == 1:
            offsets = self._d1_ov_pos_offsets
            pos_items = self._d1_ov_pos_items
            has_pos = self._d1_ov_has_pos
            n_items = self.d1_n_items
        else:
            offsets = self._d2_ov_pos_offsets
            pos_items = self._d2_ov_pos_items
            has_pos = self._d2_ov_has_pos
            n_items = self.d2_n_items

        starts_all = offsets[overlap_users]
        counts_all = offsets[overlap_users + 1] - starts_all
        valid = has_pos[overlap_users] & (counts_all < int(n_items))
        users = overlap_users[valid]
        if users.numel() == 0:
            empty = overlap_users.new_empty(0)
            return empty, empty, empty

        starts = offsets[users]
        counts = offsets[users + 1] - starts
        rand_offset = (
            torch.rand(users.shape[0], device=users.device)
            * counts.to(torch.float)
        ).long().clamp(max=counts - 1)
        sampled_pos = pos_items[starts + rand_offset]

        n_candidates = min(64, int(n_items))
        base = torch.randint(int(n_items), (users.shape[0], 1), device=users.device)
        offsets = torch.arange(n_candidates, device=users.device).view(1, -1)
        candidates = (base + offsets) % int(n_items)
        cand_users = users.unsqueeze(1).expand_as(candidates).reshape(-1)
        cand_items = candidates.reshape(-1)
        is_pos = self._overlap_pos_membership(
            cand_users, cand_items, domain,
        ).view(users.shape[0], n_candidates)
        good = ~is_pos
        has_good = good.any(dim=1)
        users = users[has_good]
        sampled_pos = sampled_pos[has_good]
        candidates = candidates[has_good]
        good = good[has_good]
        if users.numel() == 0:
            empty = overlap_users.new_empty(0)
            return empty, empty, empty

        first_good = good.to(torch.long).argmax(dim=1)
        sampled_neg = candidates[
            torch.arange(users.shape[0], device=users.device), first_good,
        ]

        return users, sampled_pos, sampled_neg

    def _pseudo_overlap_loss(self, s1, s2, inter_d1, inter_d2,
                             d1_item_emb, d2_item_emb):

        if self._d1_partner_overlap.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        losses = [
            self._pseudo_overlap_one_side(
                inter_d1, s1, d2_item_emb,
                partner_lookup=self._d1_partner_overlap,
                partner_valid_lookup=self._d1_partner_valid,
                other_domain=2,
            ),
            self._pseudo_overlap_one_side(
                inter_d2, s2, d1_item_emb,
                partner_lookup=self._d2_partner_overlap,
                partner_valid_lookup=self._d2_partner_valid,
                other_domain=1,
            ),
        ]
        valid = [l for l in losses if l is not None]
        if not valid:
            return torch.tensor(0.0, device=self.device)
        return sum(valid) / len(valid)

    def _pseudo_overlap_one_side(self, inter_self, own_shared, other_item_emb,
                                 partner_lookup, partner_valid_lookup,
                                 other_domain):

        if inter_self.numel() == 0:
            return None

        users_self = inter_self[:, 0]
        partners_all = partner_lookup[users_self]
        valid_all = partner_valid_lookup[users_self]

        if int(valid_all.sum().item()) < 2:
            return None

        anchors = users_self[valid_all]
        partners = partners_all[valid_all]

        pos_items, pos_ok = self._sample_one_overlap_pos(partners, other_domain)
        if int(pos_ok.sum().item()) < 2:
            return None
        anchors = anchors[pos_ok]
        partners = partners[pos_ok]
        pos_items = pos_items[pos_ok]

        A = anchors.numel()
        u_e = F.normalize(own_shared[anchors], p=2, dim=1)
        p_e = F.normalize(other_item_emb[pos_items], p=2, dim=1)

        logits = (u_e @ p_e.t()) / self.align_temp


        same_partner = partners.unsqueeze(0) == partners.unsqueeze(1)
        eye = torch.eye(A, dtype=torch.bool, device=logits.device)
        false_neg_mask = same_partner & ~eye
        logits = logits.masked_fill(false_neg_mask, float("-inf"))

        labels = torch.arange(A, device=logits.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        self._maybe_refresh_pseudo_pairs()

        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb, s1, s2, p1, p2 = self.forward()

        d1_users = d1_user_emb[inter_d1[:, 0]]
        d1_items = d1_item_emb[inter_d1[:, 1]]
        d1_neg_items = d1_item_emb[neg_d1_item]
        d2_users = d2_user_emb[inter_d2[:, 0]]
        d2_items = d2_item_emb[inter_d2[:, 1]]
        d2_neg_items = d2_item_emb[neg_d2_item]

        d1_bpr = self.get_bpr_loss(d1_users, d1_items, d1_neg_items)
        d2_bpr = self.get_bpr_loss(d2_users, d2_items, d2_neg_items)
        d1_reg = self.reg_weight * self.get_bpr_reg_loss(
            inter_d1[:, 0], inter_d1[:, 1], neg_d1_item, 1,
        )
        d2_reg = self.reg_weight * self.get_bpr_reg_loss(
            inter_d2[:, 0], inter_d2[:, 1], neg_d2_item, 2,
        )

        loss_xd = self._xd_infonce_loss(
            s1, s2, inter_d1, inter_d2, d1_item_emb, d2_item_emb,
        )
        loss_ortho = self._orthogonal_loss(p1, s1) + self._orthogonal_loss(p2, s2)

        loss_shared_domain_adv = torch.tensor(0.0, device=self.device)
        if self.shared_domain_adv_weight > 0:
            loss_shared_domain_adv = self._domain_classification_loss(
                s1, s2, inter_d1, inter_d2,
                self.shared_domain_discriminator,
                grl_lambda=self.domain_grl_lambda,
            )

        loss_private_domain_cls = torch.tensor(0.0, device=self.device)
        if self.private_domain_cls_weight > 0:
            loss_private_domain_cls = self._domain_classification_loss(
                p1, p2, inter_d1, inter_d2,
                self.private_domain_discriminator,
                grl_lambda=None,
            )

        loss_lna = torch.tensor(0.0, device=self.device)
        if self.lna_weight > 0:
            l_a = self._lna_private_domain(inter_d1[:, 0], inter_d1[:, 1], p1, d1_item_emb)
            l_b = self._lna_private_domain(inter_d2[:, 0], inter_d2[:, 1], p2, d2_item_emb)
            loss_lna = l_a + l_b


        loss_pseudo = torch.tensor(0.0, device=self.device)
        if self.pseudo_overlap_weight > 0 and self.has_tag_graphs:
            loss_pseudo = self._pseudo_overlap_loss(
                s1, s2, inter_d1, inter_d2, d1_item_emb, d2_item_emb,
            )

  
        loss_shared_user = torch.tensor(0.0, device=self.device)
        if self.shared_user_nce_weight > 0:
            loss_shared_user = self._shared_user_nce_loss(
                s1, s2, inter_d1, inter_d2,
            )

        loss_shared_mmd = torch.tensor(0.0, device=self.device)
        if (
            self.shared_user_mmd_weight_A != 0
            or self.shared_user_mmd_weight_B != 0
        ):
            loss_shared_mmd_A, loss_shared_mmd_B = self._shared_user_mmd_loss_terms(
                s1, s2, inter_d1, inter_d2,
            )
            loss_shared_mmd = (
                self.shared_user_mmd_weight_A * loss_shared_mmd_A
                + self.shared_user_mmd_weight_B * loss_shared_mmd_B
            )


        loss_overlap_cos = torch.tensor(0.0, device=self.device)
        if self.shared_overlap_cos_weight > 0:
            loss_overlap_cos = self._overlap_cos_loss(s1, s2, inter_d1, inter_d2)

        loss_shared_distill = torch.tensor(0.0, device=self.device)
        if self.shared_distill_weight > 0:
            loss_shared_distill = (
                self._shared_rank_distill_loss(
                    d1_users, s1[inter_d1[:, 0]], d1_items, d1_neg_items,
                )
                + self._shared_rank_distill_loss(
                    d2_users, s2[inter_d2[:, 0]], d2_items, d2_neg_items,
                )
            ) / 2

        loss_shared_cross_bpr = torch.tensor(0.0, device=self.device)
        if self.shared_cross_bpr_weight > 0:
            d1_cross_terms, d2_cross_terms = self._shared_cross_bpr_loss_terms(
                s1, s2, inter_d1, neg_d1_item, inter_d2, neg_d2_item,
                d1_item_emb, d2_item_emb,
            )
            cross_terms = d1_cross_terms + d2_cross_terms
            if cross_terms:
                loss_shared_cross_bpr = sum(cross_terms) / len(cross_terms)


        loss_shared_bpr = torch.tensor(0.0, device=self.device)
        if self.shared_bpr_weight > 0:
            loss_shared_bpr = (
                self.get_bpr_loss(s1[inter_d1[:, 0]], d1_items, d1_neg_items)
                + self.get_bpr_loss(s2[inter_d2[:, 0]], d2_items, d2_neg_items)
            ) / 2

        loss_u_sparse = d1_bpr.new_zeros(())
        loss_i_sparse = d1_bpr.new_zeros(())
        if self.has_tag_graphs:
            loss_u_sparse = (
                self.d1_u_tag_graph.sparsity_loss()
                + self.d2_u_tag_graph.sparsity_loss()
            )
            loss_i_sparse = (
                self.d1_i_tag_graph.sparsity_loss()
                + self.d2_i_tag_graph.sparsity_loss()
            )

        loss = (
            (d1_bpr + d2_bpr) / 2
            + (d1_reg + d2_reg) / 2
            + self.lambda_u_tag_sparse * loss_u_sparse
            + self.lambda_i_tag_sparse * loss_i_sparse
            + self.align_weight * loss_xd
            + self.ortho_weight * loss_ortho
            + self.shared_domain_adv_weight * loss_shared_domain_adv
            + self.private_domain_cls_weight * loss_private_domain_cls
            + self.lna_weight * loss_lna
            + self.pseudo_overlap_weight * loss_pseudo
            + self.shared_user_nce_weight * loss_shared_user
            + loss_shared_mmd
            + self.shared_overlap_cos_weight * loss_overlap_cos
            + self.shared_distill_weight * loss_shared_distill
            + self.shared_cross_bpr_weight * loss_shared_cross_bpr
            + self.shared_bpr_weight * loss_shared_bpr
        )

        return loss

    def _eval_domain(self, eval_set, user_emb, item_emb, domain_name, mode, history):
        def score_fn(users):
            return user_emb[users] @ item_emb.t()
        return full_rank_eval(eval_set, score_fn, history, domain_name)

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb, *_ = self.forward()
        if domain == 1:
            return self._eval_domain(eval_set_1, d1_user_emb, d1_item_emb, self.d1_domain_name, mode, mask_d1)
        if domain == 2:
            return self._eval_domain(eval_set_2, d2_user_emb, d2_item_emb, self.d2_domain_name, mode, mask_d2)
        d1_metrics = self._eval_domain(eval_set_1, d1_user_emb, d1_item_emb, self.d1_domain_name, mode, mask_d1)
        d2_metrics = self._eval_domain(eval_set_2, d2_user_emb, d2_item_emb, self.d2_domain_name, mode, mask_d2)
        return [d1_metrics, d2_metrics]
