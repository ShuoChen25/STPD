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
    """Hierarchical Tag LightGCN subgraph (anchors are users or items).

    Node ordering: ``[anchors (n_anchors) | tag1 (n_tag1) | tag2 (n_tag2)]``.

    Three topologies are supported, selected by ``topology``:

    ``chain`` (legacy)
        ``anchor - Tag1 - Tag2``. Tag2 sits two hops from the anchors.
    ``parallel``
        ``anchor - Tag1`` and ``anchor - Tag2`` as two independent one-hop
        channels; the Tag1-Tag2 edges are dropped from propagation.
    ``swapped``
        ``anchor - Tag2 - Tag1``. This swaps the two tag levels used by graph
        propagation: shared Tag2 is one hop from the anchors and domain-specific
        Tag1 is two hops away. The RQ-VAE codes and embeddings are not relabelled.

    The parallel topology exists because Tag1 and Tag2 are nearly independent
    (normalised MI ~0.02-0.05), which makes the Tag1-Tag2 bipartite graph
    almost complete. Routing Tag2 through it washes the per-anchor Tag2
    signal out to a near-constant vector (measured: 96% of each user's Tag2
    message is the global mean, and the per-user Tag2 distribution sits only
    0.13 total-variation away from the population average versus 0.83 for the
    direct edges). Chaining also leaks that constant back into the Tag1
    channel, so the two levels are kept on separate one-hop paths instead.

    Edges:
      - Anchor-Tag1 (``A-Tag1``): per (anchor, tag1) pair. Edge weight =
        ``sigmoid(logit)`` with one learnable logit per pair. When
        ``freq_init=True`` the logit is initialised as
        ``base + gamma * freq_score(a, t)`` (``freq_score`` = log1p of the
        (a, t) interaction count, divided by its max so it lives in
        ``[0, 1]``). When ``freq_init=False`` every edge logit is
        initialised to the constant ``base`` (used for the I-Tag1 graph).
      - Anchor-Tag2 (``A-Tag2``, ``parallel``/``swapped``): same gated form,
        per (anchor, tag2) pair.
      - Tag1-Tag2 (``chain``/``swapped``): one undirected edge per item,
        mapping the item's first-level code to its second-level code. Weight is
        fixed to 1.

    Tag1 (per-domain mid-tag) and Tag2 (outer-tag) embeddings are owned by
    the parent module and passed in at :meth:`propagate` time, so U-Tag and
    I-Tag graphs can keep independent tag parameters while reusing the same
    graph kernel. Symmetric LightGCN normalisation uses the binary degree of
    each node (counting multiplicities for repeated Tag1-Tag2 edges).
    """

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
        """Both directions of a gated edge set, sharing one logit per pair."""
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
            # The reversed chain starts at Tag2. Keep the constructor API
            # uniform, but intentionally remove all direct Anchor-Tag1 edges.
            at1_edges = at1_edges.new_zeros((0, 2))

        # --- Anchor-Tag1 learnable gates ---
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

        # --- Anchor-Tag2 learnable gates (parallel/swapped topologies) ---
        # Tag2 is the cross-domain bridge, so a direct one-hop edge keeps each
        # anchor's own Tag2 profile instead of the Tag1-averaged version.
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

        # --- Tag1-Tag2 edges (one per item, weight 1; chain topologies only) ---
        if topology in ("chain", "swapped"):
            t1_node = t1t2_edges[:, 0].long() + n_anchors
            t2_node = t1t2_edges[:, 1].long() + n_anchors + n_tag1
            t_row = torch.cat([t1_node, t2_node])
            t_col = torch.cat([t2_node, t1_node])
        else:
            t_row = torch.zeros(0, dtype=torch.long)
            t_col = torch.zeros(0, dtype=torch.long)

        # --- Static (fixed-weight) degree contribution from Tag1-Tag2 edges ---
        # Weighted LightGCN: deg_i = sum_j w_ij. Tag1-Tag2 edges have w=1 and
        # are fixed, so their contribution can be cached; the A-Tag1 part
        # (w_ij = sigmoid(logit)) is recomputed each forward from the current
        # gates so that normalisation always matches the learnable edge weights.
        deg_static = torch.zeros(self.n_total, dtype=torch.float32)
        deg_static.scatter_add_(0, t_row, torch.ones_like(t_row, dtype=torch.float32))

        # Raw (tag1, tag2) pairs without node offsets, used by downstream
        # consumers that need to map Tag1 ids onto Tag2 ids (e.g. computing
        # per-user Tag2 distributions).
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
        # ``_edge_values`` must emit values in exactly this block order.
        self.register_buffer(
            "row", torch.cat([at_row, at2_row, t_row]), persistent=False,
        )
        self.register_buffer(
            "col", torch.cat([at_col, at2_col, t_col]), persistent=False,
        )
        self.register_buffer("deg_static", deg_static.to(dtype=dtype), persistent=False)
        self.n_at = n_at

    def _edge_values(self):
        """Weighted symmetric normalisation shared by all propagation modes.

        ``deg_i = sum_j w_ij`` is recomputed from the current gates so the
        normaliser follows the learnable edge weights (weighted LightGCN).
        """
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
        # Order must match the ``row``/``col`` buffers built in __init__.
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
        """Propagate and return the three additive source contributions.

        Propagation is linear in the node features, so seeding three disjoint
        feature blocks with one source each and running a single pass yields
        the exact per-source decomposition of the anchor output:

            ``propagate(...) == base + tag1_path + tag2_path``

        (exact in eval mode; in train mode the dropout masks differ per block,
        which is distributionally equivalent.) This lets the caller route the
        Tag2 contribution -- the cross-domain bridge -- into the shared user
        branch, so the shared-branch losses backpropagate through Tag2 instead
        of bypassing it.
        """
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
        """Weighted mean of sigmoid gates — drives Anchor-Tag edges toward sparsity.

        Each edge's sigmoid gate is multiplied by its (non-trainable)
        ``at*_sparsity_weights`` entry before averaging. When the buffers are
        left at their default (all ones) this reduces to a plain mean. In the
        ``parallel`` and ``swapped`` topologies the Anchor-Tag2 gates are
        averaged in as well.
        """
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
        # U-I LightGCN depth is decoupled from the Tag-graph depth so the two
        # propagation stages can be tuned independently. Falls back to
        # ``intra_domain_layers`` when ``ui_graph_layers`` is absent from the
        # config (preserves legacy behaviour).
        self.ui_graph_layers = int(
            params.get("ui_graph_layers", self.intra_domain_layers)
        )
        self.n_shared_users = args.data["n_shared_users"]
        self.d1_n_users, self.d2_n_users = args.d1["n_users"], args.d2["n_users"]
        self.d1_n_items, self.d2_n_items = args.d1["n_items"], args.d2["n_items"]
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        # Raw U-I LightGCN inputs used when tag graphs are unavailable, and
        # by the ``wocross`` ablation. In the normal STPD tag path, the
        # U-I graph receives the U/I-Tag graph outputs instead.
        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        # Dedicated U / I inputs for the U-Tag and I-Tag graphs. The I-Tag item
        # tables are initialised from the offline PCA-reduced content vectors
        # after the generic module initialisation below, then trained normally.
        self.d1_user_tg = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_user_tg = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_tg = nn.Embedding(
            self.d1_n_items, self.emb_dim, dtype=self.dtype
        )
        self.d2_item_tg = nn.Embedding(
            self.d2_n_items, self.emb_dim, dtype=self.dtype
        )

        self.dropout = nn.Dropout(self.drop_rate)

        # Gate MLPs for decomposing user embeddings into domain-private/shared
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

        # Fusion MLPs: input = concat(private, shared), output = shared fusion weight
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

        # CSR-style per-overlap-user positive-item lookup, used by the
        # cross-domain InfoNCE alignment loss so that an overlap user only
        # needs to appear in ONE domain's batch — we sample their positive
        # item in the OTHER domain from this training-set cache instead of
        # requiring the same user to also be present in the other domain's
        # batch (critical in the sparse-overlap setting).
        self._build_overlap_pos_lookup(args)

        # All-user CSR pos-item lookup (every user, not just overlap), used
        # by the pseudo-overlap multi-positive InfoNCE so we can fetch the
        # full training-set positives of an arbitrary partner user in the
        # other domain (the partner can be overlap, d1-only, or d2-only).
        self._build_all_pos_lookup(args)

        # Unified xavier initialization for all trainable modules.
        self.apply(self._init_weights)

        # Replace the random I-Tag item-table rows with their offline PCA
        # content embeddings. This must happen after ``self.apply`` so the
        # generic Xavier pass does not overwrite the PCA values.
        self._maybe_load_pca_item_init(args)

        # Match teacher pre-training initialisation: items use xavier_uniform_
        # with gain=1 (see pretrain_teacher.py); users use xavier_normal_ which
        # is already applied by _init_weights above.
        nn.init.xavier_uniform_(self.d1_item.weight, gain=1)
        nn.init.xavier_uniform_(self.d2_item.weight, gain=1)

        # Ablation: freeze PCA content as the I-Tag anchor and learn only a
        # 128x128 projection, so no per-item free parameter exists and the
        # Tag graph's cross-item smoothing becomes a real constraint that
        # cannot be row-wise absorbed by the anchor table.
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

        # How the private/shared user split is produced. In `tag2` mode the
        # shared branch IS the Tag2 contribution of the Tag graph, so the
        # heavily weighted shared-branch losses (shared_cross_bpr, shared MMD)
        # backpropagate through the shared Tag2 bridge rather than through a
        # free gated vector that bypasses every Tag parameter.
        self.shared_split_mode = str(
            args.params.get("shared_split_mode", "gate")
        ).strip().lower()
        if self.shared_split_mode not in ("gate", "tag2"):
            raise ValueError(
                f"shared_split_mode must be 'gate' or 'tag2', got "
                f"{self.shared_split_mode!r}."
            )
        if self.shared_split_mode == "tag2" and not self.use_tag_outputs_as_ui_input:
            # The wocross ablation bypasses the Tag path entirely, so there is
            # no Tag2 channel to derive the shared branch from. Fall back to
            # the gate split so the STPD/wocross pair can share one config.
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

        # Tag-graph topology. `chain` is the legacy anchor-Tag1-Tag2 cascade;
        # `parallel` makes Tag1/Tag2 independent one-hop channels; `swapped`
        # reverses the cascade to anchor-Tag2-Tag1 for a controlled ablation.
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

        # Keep fallback/wocross U-I embeddings random at initialisation:
        # users from _init_weights, items from the xavier_uniform_ calls above.

        # Neutral init for gate and fusion: g ~= 0.5 and alpha ~= 0.5 so that
        # private / shared start at equal magnitude and contribute equally to
        # the fused user, giving the cross-domain branch a visible signal
        # from step 0 instead of starting in a private-dominant regime.
        nn.init.constant_(self.gate_d1[-1].bias, 0.0)
        nn.init.constant_(self.gate_d2[-1].bias, 0.0)
        nn.init.constant_(self.fuse_d1[-1].bias, 0.0)
        nn.init.constant_(self.fuse_d2[-1].bias, 0.0)

        # Build per-domain U-Tag and I-Tag graphs from the RQ-VAE codes and
        # codebooks. U-Tag and I-Tag keep independent Tag parameters.
        self.has_tag_graphs = False
        self._build_tag_graphs(args)

        # Optionally transfer the teacher's learned Tag-branch anchor
        # embeddings and U-Tag1 / I-Tag1 edge gates. Tag embeddings remain
        # RQ-VAE initialised.
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

        # Lna: in-batch contrastive loss on domain-private user vectors.
        # For each (user, pos_item) in the batch, push the score between the
        # user's domain-private vector and its positive item away from the
        # scores against every other in-batch positive item.
        self.lna_weight = params.get("lna_weight", 0.0)
        self.lna_temp = params.get("lna_temp", 0.2)

        # Pseudo-overlap alignment via Tag2 profile similarity.
        # Every user u (overlap and non-overlap) is mapped to ONE closest
        # partner v in the other domain's OVERLAP user pool (ids in
        # ``[0, n_shared)``), by argmax of cosine similarity between their
        # Tag2 preference distributions (Tag2 is the cross-domain shared
        # vocabulary, so distributions are directly comparable). Self-match
        # is allowed: if u is
        # itself an overlap user, its closest is typically u in the other
        # domain, so the partner-derived positive is just u's own real
        # cross-domain history.
        # The loss is a STANDARD 1-positive InfoNCE on the shared branch:
        # for each batch anchor we draw one random training positive from
        # its partner's other-domain history; in-batch logits form a
        # [B, B] matrix where the diagonal is the positive and off-diagonal
        # entries are negatives (other anchors' partner items). Anchors
        # sharing the same partner are masked out as false negatives.
        # Reuses ``align_temp`` for the temperature.
        self.pseudo_overlap_weight = params.get("pseudo_overlap_weight", 0.0)
        self.pseudo_overlap_refresh_every = int(
            params.get("pseudo_overlap_refresh_every", 3)
        )

        # User-user InfoNCE on the domain-shared vectors. OVERLAP-ONLY anchors.
        # Anchors are overlap users (``u < n_shared``) appearing in the
        # current batch; positive = ``s_other[u]`` (same id in the other
        # domain); candidate pool = all OTHER-domain overlap users plus
        # OTHER-domain batch users. Symmetric across d1/d2; reuses
        # ``align_temp``.
        self.shared_user_nce_weight = params.get("shared_user_nce_weight", 0.0)

        # Directional distribution-level (MMD) alignment between batch
        # users' shared vectors across the two domains. If the legacy
        # symmetric knob is non-zero, both directions use that value;
        # otherwise domain-specific A/B weights are used.
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

        # Paired overlap-user cosine alignment: only pulls s1[u] and s2[u]
        # together for the same overlap identity, with no contrastive
        # negatives.
        self.shared_overlap_cos_weight = params.get("shared_overlap_cos_weight", 0.0)

        # Shared-channel ranking distillation: the shared vector learns the
        # final fused user's pos-vs-neg margin, but the teacher margin and
        # item vectors are detached so this mainly trains the shared user
        # channel.
        self.shared_distill_weight = params.get("shared_distill_weight", 0.0)

        # Original standalone shared-channel BPR. This directly asks s1/s2
        # to rank each domain's own positive above its sampled negative.
        self.shared_bpr_weight = params.get("shared_bpr_weight", 0.0)

        # Shared-channel cross-domain BPR, controlled by one symmetric
        # weight for both source-domain shared branches.
        self.shared_cross_bpr_weight = params.get("shared_cross_bpr_weight", 0.0)

        # Epoch tracker compatible with main.py (`args.model._epoch[0] = i`).
        self.register_buffer(
            "_epoch", torch.zeros(1, dtype=torch.long), persistent=False,
        )
        self._last_pseudo_refresh = -1

        # Cached closest-overlap-partner lookup (refilled every
        # ``pseudo_overlap_refresh_every`` epochs inside ``calculate_loss``).
        # ``_d1_partner_overlap[u]`` is the local-d2 overlap-user id (i.e.
        # in ``[0, n_shared)``) closest to local-d1 user u; symmetric for
        # ``_d2_partner_overlap``. ``_*_partner_valid[u]`` is True iff that
        # partner has a non-empty other-domain training history (so a
        # positive can be sampled).
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

    # ---- Initialization ----

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
        """Initialise the normal STPD I-Tag item tables from PCA vectors."""
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
        """Return the full [n_items, emb_dim] I-Tag anchor tensor.

        In ``free`` mode this is the trainable ``d{1,2}_item_tg.weight``.
        In ``content_proj`` mode the frozen PCA content is projected through
        a learnable 128x128 matrix, so no per-item free parameter exists.
        """
        if self.item_anchor_mode == "content_proj":
            proj = self.d1_item_proj if domain == 1 else self.d2_item_proj
            table = self.d1_item_tg if domain == 1 else self.d2_item_tg
            return proj(table.weight)
        return (self.d1_item_tg if domain == 1 else self.d2_item_tg).weight

    def _random_tag_embedding(self, n_tags):
        """Create a Xavier-initialised Tag embedding matrix on CPU."""
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
        """Return ``(edges, freqs)`` for U-Tag in this domain.

        ``edges`` is ``[E, 2]`` of unique ``(global_user, tag_id)`` pairs;
        ``freqs`` is ``[E]`` interaction counts. ``code_col`` selects the RQ
        level: ``0`` for the per-domain Tag1, ``1`` for the shared Tag2 (used
        by the ``parallel``/``swapped`` direct U-Tag2 edges). Domain-2
        non-shared users are remapped to the global id space.
        """
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
        """Per-global-user Tag2 interest profile from interacted items.

        This mirrors the RQ-VAE overlap-user alignment objective: a user's
        Tag2 distribution is the average Tag2 assignment of its interacted
        items.  New RQ-VAE artifacts save soft item->Tag2 assignments; older
        artifacts only have hard RQ codes, so they fall back to one-hot counts.
        """
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
        """Load RQ-VAE artifacts and build per-domain U-Tag and I-Tag graphs.

        U-Tag and I-Tag graphs keep independent role parameters. Cross-domain
        Tag1 sharing follows the RQ-VAE artifact; Tag2 is always shared.
        Each graph has learnable gates on whichever tag level is directly
        attached to its anchors (Tag1 normally, Tag2 for ``swapped``).
        """
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
        d1_codes = rq["d1_item_rqcodes"].long()   # [d1_n_items, 2] -> (idx1, idx2)
        d2_codes = rq["d2_item_rqcodes"].long()
        d1_cb1 = rq["d1_codebook1"]               # [d1_n_tag1, latent_dim]
        d2_cb1 = rq["d2_codebook1"]               # [d2_n_tag1, latent_dim]
        shared_cb2 = rq["shared_codebook2"]       # [n_tag2, latent_dim]
        d1_item_tag2_soft = rq.get("d1_item_tag2_soft")
        d2_item_tag2_soft = rq.get("d2_item_tag2_soft")

        # Ablation: shuffle item->code assignments per domain. Row permutation
        # preserves cluster sizes, the Tag1-Tag2 edge multiset, and the shared
        # Tag2 semantics (code k means the same in both domains), but breaks
        # the semantic connection between item identity and its cluster.
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

        # Role-specific Tag embeddings. The RQ codes always define graph
        # topology; this option only ablates the codebook-vector initialisation.
        # U-Tag and I-Tag start from identical values, as in the RQ-VAE path,
        # but remain independent trainable parameters afterwards.
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

        # I-Tag1 edges: one per item, linking item id to its own 1st-level code.
        d1_it_edges = torch.stack(
            [torch.arange(self.d1_n_items, dtype=torch.long), d1_codes[:, 0]], dim=1,
        )
        d2_it_edges = torch.stack(
            [torch.arange(self.d2_n_items, dtype=torch.long), d2_codes[:, 0]], dim=1,
        )

        # Direct Anchor-Tag2 edges for the ``parallel`` and ``swapped``
        # topologies. In ``swapped`` these replace the direct Anchor-Tag1 edges.
        # Items map to their own 2nd-level code; users accumulate the Tag2
        # codes of the items they interacted with, weighted by frequency.
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

        # Per-item Tag1-Tag2 edges: codes column 0 is Tag1 id, column 1 is Tag2 id.
        d1_t1t2 = torch.stack([d1_codes[:, 0], d1_codes[:, 1]], dim=1)
        d2_t1t2 = torch.stack([d2_codes[:, 0], d2_codes[:, 1]], dim=1)

        # Popularity-weighted sparsity copied from teacher pre-training:
        # I-Tag edges are weighted by item popularity; U-Tag edges by the
        # aggregate popularity of the target Tag1.
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

        # Same popularity weighting for the direct Anchor-Tag2 edges: items by
        # their own popularity, users by the aggregate popularity of the Tag2.
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
        """Copy teacher Tag-branch anchor embeddings and edge gates.

        Teacher U/I embeddings are loaded only into ``d{1,2}_{user,item}_tg``
        so the U/I-Tag branch starts from the pre-trained Tag graph and then
        feeds the U-I LightGCN. Role-specific Tag1/Tag2 parameters stay at
        their RQ-VAE codebook initialisation.
        """
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
            # ---- Tag-branch anchor embeddings ----
            # In content_proj mode the I-Tag item table is frozen PCA content;
            # skip teacher transfer for it so the content anchor is preserved.
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

            # ---- Anchor-Tag1 edge logits ----
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

    # ---- Graph Construction ----

    def _build_norm_graph(self, sp_mat):
        norm = symmetric_normalize(sp_mat)
        return convert_sp_mat_to_sp_tensor(norm, dtype=self.dtype).to(self.device)

    def _build_overlap_pos_lookup(self, args):
        """Build CSR ``(offsets, items)`` of training positives per overlap user.

        For each overlap user id ``u`` in ``[0, n_shared_users)`` we collect
        the set of items they interacted with in d1 and d2 from
        ``args.d1["train"]`` / ``args.d2["train"]``. Stored as registered
        buffers so they follow the model to GPU and survive ``.cuda()``:

        - ``_d1_ov_pos_offsets`` : LongTensor ``[n_shared_users + 1]``
        - ``_d1_ov_pos_items``   : LongTensor ``[E1]`` (flat item ids)
        - ``_d1_ov_has_pos``     : BoolTensor ``[n_shared_users]``
        - same triple for d2

        Sampling a random d1 positive for anchor users ``u`` is then a few
        fully-vectorised ops on the GPU.
        """
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
        """CSR ``(offsets, items)`` of training positives for EVERY user.

        Same layout as :meth:`_build_overlap_pos_lookup`, but keyed by
        local-domain user id over the FULL ``[0, dX_n_users)`` range. Used
        by the pseudo-overlap multi-positive InfoNCE to fetch the full
        positive list of an arbitrary partner user in the other domain.
        """
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
        """Gather full training-positive list per user (any user id).

        Returns ``(items, mask, valid_anchor)``:

        - ``items``         : LongTensor ``[A, max_count]``, item ids in
          the target ``domain``. Padding cells are zero-filled.
        - ``mask``          : BoolTensor ``[A, max_count]``, True where
          ``items[i, j]`` is a real training positive of ``users[i]``.
        - ``valid_anchor``  : BoolTensor ``[A]``, True iff the user has at
          least one positive in this domain.

        Reads the all-user lookup tables, so ``users`` may include d1-only
        or d2-only ids in their respective local id spaces.
        """
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

    # ---- LightGCN Aggregation ----

    def aggregate(self, graph, all_emb, n_layers):
        embeddings = [all_emb]
        for _ in range(n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            all_emb = self.dropout(all_emb)
            embeddings.append(all_emb)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    # ---- Split Helpers ----

    def _split_domain_ui(self, emb, domain):
        if domain == 1:
            return emb[:self.d1_n_users], emb[self.n_users:]

        user = torch.cat([emb[:self.n_shared_users], emb[self.d1_n_users:self.n_users]], dim=0)
        return user, emb[self.n_users:]

    # ---- Forward ----

    def forward(self):
        # Tag-graph stage: propagate anchor U / I embeddings through independent
        # U-Tag1-Tag2-Tag1-U and I-Tag1-Tag2-Tag1-I subgraphs. U-Tag and
        # I-Tag use separate role parameters; cross-domain sharing follows the
        # selected RQ-VAE codebook pattern.
        split_on_tag2 = (
            self.shared_split_mode == "tag2" and self.has_tag_graphs
        )
        d1_user_t2 = d2_user_t2 = None
        d1_item_t2 = d2_item_t2 = None

        if self.has_tag_graphs and split_on_tag2:
            # Decomposed propagation: keep the Tag2 contribution separately so
            # it can become the shared user branch. The three parts sum to the
            # same output the non-decomposed path produces.
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
            # The propagated U/I anchors become the next U-I LightGCN input.
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

        # Per-domain U-I LightGCN now takes the U/I-Tag outputs as its input.
        # If tag graphs are disabled, fall back to the raw U/I embeddings.
        # ``aggregate`` is linear, so in tag2-split mode the Tag2 channel is
        # pushed through the same graph alongside the full channel (batched on
        # the feature axis) to recover the Tag2 component of the final output.
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

        # Decompose user embeddings into domain-private and domain-shared.
        if split_on_tag2:
            # Shared branch IS the Tag2 contribution, so shared_cross_bpr and
            # the shared MMD backpropagate into the shared Tag2 bridge. The
            # ``p + s == user_raw`` invariant of the gate path is preserved.
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

        # Fuse private and shared via gating: alpha is the shared weight
        alpha1 = torch.sigmoid(self.fuse_d1(torch.cat([p1, s1], dim=-1)))
        d1_user = alpha1 * s1 + (1 - alpha1) * p1
        alpha2 = torch.sigmoid(self.fuse_d2(torch.cat([p2, s2], dim=-1)))
        d2_user = alpha2 * s2 + (1 - alpha2) * p2

        return d1_user, d1_item, d2_user, d2_item, s1, s2, p1, p2

    def _xd_infonce_loss(self, s1, s2, inter_d1, inter_d2,
                         d1_item_emb, d2_item_emb):
        """Overlap-only item InfoNCE on the shared branch.

        Only overlap users (``u < n_shared``) in the current batch act as
        anchors. For each anchor we build two standard 1-positive InfoNCE
        terms:

          - self-domain positive: the positive item carried by this training
            sample, e.g. ``d1_item[i+]`` for a d1-side anchor.
          - other-domain positive: one random training positive of the same
            overlap user in the other domain.

        The negative pool is the union of all positive items appearing in
        both domains' current batch, following the in-batch style of Lna.
        The anchor's own positive column is masked out of the negative pool
        for its corresponding 1-positive term.
        """
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
            # d1-side anchors: s1[u], self item = d1, other item = sampled d2.
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
            # d2-side anchors: s2[u], self item = d2, other item = sampled d1.
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
        """One direction of the overlap-only item InfoNCE."""
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

        # Sample one cross-domain training positive per overlap user; drop
        # anchors whose same user has empty other-domain history.
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
        """Standard 1-positive InfoNCE with an explicit positive column."""
        pos_logits = (anchor_e * pos_e).sum(dim=1, keepdim=True) / self.align_temp
        neg_logits = (anchor_e @ neg_e.t()) / self.align_temp
        if neg_mask is not None:
            neg_logits = neg_logits.masked_fill(neg_mask, float("-inf"))
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(anchor_e.shape[0], dtype=torch.long, device=anchor_e.device)
        return F.cross_entropy(logits, labels)

    def _orthogonal_loss(self, p, s):
        """Orthogonality loss between private and shared vectors."""
        p_norm = F.normalize(p, dim=-1)
        s_norm = F.normalize(s, dim=-1)
        return (p_norm * s_norm).sum(dim=-1).pow(2).mean()

    def _domain_classification_loss(self, z1, z2, inter_d1, inter_d2,
                                    discriminator, grl_lambda=None):
        """Domain CE over current-batch users; optional GRL makes it adversarial."""
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
        """User-user InfoNCE on the domain-shared vectors. OVERLAP-ONLY anchors.

        Anchors are overlap users (``u < n_shared``) -- the only users
        for whom a cross-domain positive ``s_other[u]`` is well-defined.
        For each anchor ``u`` appearing in one domain's batch, the positive
        remains ``s_other[u]`` while the candidate pool is the union of:

          * all OTHER-domain overlap users ``s_other[:n_shared]``
          * the OTHER-domain users appearing in the current batch

        The positive is included once in the denominator and the label is
        its column in this mixed candidate pool. Non-overlap users are never
        anchors, but OTHER-domain non-overlap batch users can be negatives.

        Symmetric across d1/d2; cosine similarity (L2-normalised);
        reuses ``align_temp`` as the temperature. Returns 0 when no
        overlap users exist or neither side has any overlap anchors in
        the batch.
        """
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
        """One direction of :meth:`_shared_user_nce_loss`.

        ``batch_users`` are local ids in ``s_self``'s domain. Anchors
        are restricted to overlap users (``u < n_shared``) because only
        they have a well-defined cross-domain positive ``s_other[u]``.
        The candidate pool is all overlap users plus the other domain's
        in-batch users, deduplicated in local id space.
        """
        unique_users = torch.unique(batch_users)
        if unique_users.numel() == 0:
            return None

        overlap_mask = unique_users < self.n_shared_users
        if not overlap_mask.any():
            return None
        anchors = unique_users[overlap_mask]

        a = F.normalize(s_self[anchors], p=2, dim=1)         # [A, d]
        overlap_ids = torch.arange(self.n_shared_users, device=batch_users.device)
        candidate_ids = torch.unique(
            torch.cat([overlap_ids, other_batch_users]), sorted=True,
        )
        b = F.normalize(s_other[candidate_ids], p=2, dim=1)  # [C, d]
        logits = (a @ b.t()) / self.align_temp               # [A, C]

        labels = torch.searchsorted(candidate_ids, anchors)
        return F.cross_entropy(logits, labels)

    def _shared_user_mmd_loss_terms(self, s1, s2, inter_d1, inter_d2):
        """Directional MMD alignment between batch users' shared vectors.

        Computes a multi-bandwidth RBF MMD between the SHARED vectors of
        the d1-batch users and the d2-batch users. Unlike the per-user
        InfoNCE in :meth:`_shared_user_nce_loss`, this term needs no
        cross-domain identity correspondence -- ALL batch users (overlap
        and non-overlap) participate. The returned terms apply stop-gradient
        on the opposite domain, so domain A and B can have independent loss
        weights while still being pulled toward each other's distribution.

        Robustness to imbalanced batches:
          * MMD's biased estimator is asymptotically unbiased with
            normalisers ``1/n^2``, ``1/m^2``, ``1/(nm)`` baked in -- so
            ``n != m`` is handled by construction.
          * When either side has fewer than ``MIN_USERS`` unique users,
            the K_yy / K_xx variance dominates, so we skip rather than
            inject noise.
          * Inputs are L2-normalised to make sigma selection robust to
            absolute embedding scale.
          * Multi-bandwidth (sum over a fixed sigma grid) reduces
            sensitivity to a single bandwidth choice.
        """
        if not (inter_d1.numel() and inter_d2.numel()):
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        u1 = torch.unique(inter_d1[:, 0])
        u2 = torch.unique(inter_d2[:, 0])

        # Guard against high-variance MMD estimates when either side has
        # too few unique users (Copenhagen's joint sampling can produce
        # heavily imbalanced d1/d2 splits when interaction counts differ).
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
        """Cosine alignment for overlap users seen by either domain batch."""
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
        """Multi-bandwidth RBF MMD^2 between two unmatched sample sets.

        Uses the squared-distance form
            ``k(a,b) = exp(-||a-b||^2 / (2 sigma^2))``
        and averages over the supplied bandwidths. ``x`` and ``y`` may
        have different leading dimensions; the .mean() over each kernel
        block handles the n^2 / m^2 / (n*m) normalisation automatically.
        """
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
        """In-batch contrastive loss on domain-private user vectors.

        For each batch sample ``(u, i+)``, treat the scores between ``u``'s
        private vector and every other in-batch positive item as negatives,
        and push the diagonal (the user's own positive) above them via a
        temperature-scaled softmax cross-entropy.
        """
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
        """Match shared-branch ranking margins to the fused-user teacher."""
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
        """BPR for overlap users against the same user's other-domain items."""
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
        """Return d1/s1 and d2/s2 shared/cross BPR loss terms."""
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
        """Recompute every-anchor -> closest-overlap-partner mapping every
        ``pseudo_overlap_refresh_every`` epochs.

        Partner pool is restricted to OVERLAP users only (local ids
        ``[0, n_shared)`` in the other domain): their other-domain history
        is real interaction data, which gives high-quality positives.

        For an overlap anchor (id ``< n_shared``) the closest is typically
        itself in the other domain (same global user, similar Tag2
        profile), and self-match is ALLOWED.

        Result buffers:
            ``_d1_partner_overlap`` : ``[d1_n_users]`` long, values in
                ``[0, n_shared)`` -- closest overlap user in d2 by Tag2
                profile cosine.
            ``_d2_partner_overlap`` : symmetric for d2-side anchors.
            ``_*_partner_valid``    : True iff that partner has a
                non-empty training history in the OTHER domain.
        """
        if not self.training or not self.has_tag_graphs:
            return
        # Refresh whenever a consumer of the pseudo-overlap partner lookups
        # is active: currently only the item-positive pseudo-overlap loss.
        # The shared-vector user-user InfoNCE no longer relies on Tag2
        # partner buffers (it operates purely on overlap-user ids).
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
            # No overlap users -> no candidate partner pool exists.
            self._d1_partner_overlap = torch.zeros(self.d1_n_users, dtype=torch.long, device=device)
            self._d2_partner_overlap = torch.zeros(self.d2_n_users, dtype=torch.long, device=device)
            self._d1_partner_valid = torch.zeros(self.d1_n_users, dtype=torch.bool, device=device)
            self._d2_partner_valid = torch.zeros(self.d2_n_users, dtype=torch.bool, device=device)
            print(f"\n[Pseudo-overlap] Epoch {cur}: n_shared=0, partner pool empty -> disabled.")
            return

        profile_d1 = self._d1_user_tag2_profile.float()
        profile_d2 = self._d2_user_tag2_profile.float()

        # d1-side anchors: local d1 ids [0, d1_n_users) == global ids.
        p1 = profile_d1[:self.d1_n_users]                                     # [d1_n_users, n_tag2]
        # d2-side anchors: local d2 ids -> global ids.
        d2_global_ids = torch.cat([
            torch.arange(n_s, device=device),
            torch.arange(self.d1_n_users, self.n_users, device=device),
        ])
        p2 = profile_d2[d2_global_ids]                                        # [d2_n_users, n_tag2]

        # Overlap-only candidate pools (first n_s rows of each).
        p1_ov = profile_d1[:n_s]                                              # d1-side perspective of overlap users
        p2_ov = profile_d2[:n_s]                                              # d2-side perspective of overlap users

        z1 = F.normalize(p1, p=2, dim=-1)
        z2 = F.normalize(p2, p=2, dim=-1)
        z1_ov = F.normalize(p1_ov, p=2, dim=-1)
        z2_ov = F.normalize(p2_ov, p=2, dim=-1)

        # d1 anchor -> closest d2-overlap (in [0, n_s)).
        sim_d1 = z1 @ z2_ov.t()                                               # [d1_n_users, n_s]
        d1_partner = sim_d1.argmax(dim=1).long()                              # [d1_n_users]
        d1_best = sim_d1.gather(1, d1_partner.unsqueeze(1)).squeeze(1)

        # d2 anchor -> closest d1-overlap (in [0, n_s)).
        sim_d2 = z2 @ z1_ov.t()                                               # [d2_n_users, n_s]
        d2_partner = sim_d2.argmax(dim=1).long()                              # [d2_n_users]
        d2_best = sim_d2.gather(1, d2_partner.unsqueeze(1)).squeeze(1)

        # Validity: partner must have a non-empty OTHER-domain training set.
        # d1 anchor's partner is an overlap user; its OTHER-domain history is
        # in d2 -> use ``_d2_ov_has_pos[partner]``.
        d1_valid = self._d2_ov_has_pos[d1_partner].clone()
        d2_valid = self._d1_ov_has_pos[d2_partner].clone()

        self._d1_partner_overlap = d1_partner
        self._d2_partner_overlap = d2_partner
        self._d1_partner_valid = d1_valid
        self._d2_partner_valid = d2_valid

        # Diagnostic: how many overlap anchors are matched to themselves,
        # and the sim distribution of all anchors' best partners.
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
        """Random-sample ONE training positive per overlap user.

        ``overlap_users`` is a long tensor of overlap-user ids in
        ``[0, n_shared)``; returns ``(pos_items, valid)`` where
        ``valid[i]`` is True iff that overlap user has at least one
        training positive in ``domain``. Items for invalid rows are
        arbitrary (the caller masks them out via ``valid``).
        """
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
        """Return True where ``items`` are training positives for overlap users."""
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
        """Sample one positive and one non-training negative for overlap users.

        Returns only valid users: they must have at least one training positive
        and at least one item outside their training positives in ``domain``.
        """
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
        """Standard 1-positive InfoNCE on the shared branch.

        For each anchor user u in the batch:

          1. partner v = ``_d{self}_partner_overlap[u]``                     # closest overlap user in OTHER domain
          2. pos_item i_v ~ Uniform( v's other-domain training positives )  # one random sample
          3. positive logit  = cos(s_{self}[u], item_emb[i_v]) / align_temp
          4. negative logits = cos(s_{self}[u], item_emb[i_{v'}]) for every
             OTHER anchor u' in the same batch                              # in-batch negatives
          5. Anchors u' that share a partner with u (v' == v) are masked
             off-diagonal as false negatives.

        The two directions (d1 anchors -> d2 partner items, d2 anchors ->
        d1 partner items) are computed independently and averaged.
        """
        if self._d1_partner_overlap.numel() == 0:
            # Refresh hasn't populated buffers (training=False or first call).
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
        """One direction of the 1-pos InfoNCE (see :meth:`_pseudo_overlap_loss`).

        Builds a ``[A, A]`` similarity matrix between A anchors and the A
        partner-sampled items; the i-th row's label is i. Anchors that
        share a partner with anchor i contribute false-negative entries in
        row i and are masked to ``-inf``.
        """
        if inter_self.numel() == 0:
            return None

        users_self = inter_self[:, 0]                                          # [B] local self-domain ids
        partners_all = partner_lookup[users_self]                              # [B] in [0, n_shared)
        valid_all = partner_valid_lookup[users_self]                           # [B] bool

        if int(valid_all.sum().item()) < 2:
            # Need at least 2 anchors for in-batch InfoNCE to have a negative.
            return None

        anchors = users_self[valid_all]                                        # [A]
        partners = partners_all[valid_all]                                     # [A]

        pos_items, pos_ok = self._sample_one_overlap_pos(partners, other_domain)
        # pos_ok is guaranteed True since partner_valid_lookup already requires
        # has_pos, but re-check defensively in case of state mismatch.
        if int(pos_ok.sum().item()) < 2:
            return None
        anchors = anchors[pos_ok]
        partners = partners[pos_ok]
        pos_items = pos_items[pos_ok]

        A = anchors.numel()
        u_e = F.normalize(own_shared[anchors], p=2, dim=1)                     # [A, d]
        p_e = F.normalize(other_item_emb[pos_items], p=2, dim=1)               # [A, d]

        # [A, A] in-batch logits: row i is anchor i, col j is partner-item j.
        logits = (u_e @ p_e.t()) / self.align_temp

        # False-negative mask: off-diagonal entries where two anchors share
        # the same partner (so j's positive is "the same person" as i's, not
        # a true negative). Diagonal is the positive and must stay.
        same_partner = partners.unsqueeze(0) == partners.unsqueeze(1)          # [A, A]
        eye = torch.eye(A, dtype=torch.bool, device=logits.device)
        false_neg_mask = same_partner & ~eye
        logits = logits.masked_fill(false_neg_mask, float("-inf"))

        labels = torch.arange(A, device=logits.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        # Recompute pseudo-overlap pairs at epoch boundaries (no_grad inside).
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

        # Domain disentanglement after the private/shared split:
        # shared vectors pass through GRL, private vectors do not.
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

        # Lna on domain-private vectors (in-batch contrastive over positives).
        loss_lna = torch.tensor(0.0, device=self.device)
        if self.lna_weight > 0:
            l_a = self._lna_private_domain(inter_d1[:, 0], inter_d1[:, 1], p1, d1_item_emb)
            l_b = self._lna_private_domain(inter_d2[:, 0], inter_d2[:, 1], p2, d2_item_emb)
            loss_lna = l_a + l_b

        # Pseudo-overlap alignment on shared vectors via Tag2-closest overlap partner.
        # ALL users (overlap + non-overlap) are mapped to their closest overlap
        # user in the other domain; one random training positive of that partner
        # becomes the anchor's positive in a standard in-batch 1-pos InfoNCE.
        # Same-partner anchors mask each other as false negatives.
        loss_pseudo = torch.tensor(0.0, device=self.device)
        if self.pseudo_overlap_weight > 0 and self.has_tag_graphs:
            loss_pseudo = self._pseudo_overlap_loss(
                s1, s2, inter_d1, inter_d2, d1_item_emb, d2_item_emb,
            )

        # User-user InfoNCE on shared vectors: each anchor is pulled to its
        # cross-domain partner and pushed away from all OTHER-domain overlap
        # users plus OTHER-domain in-batch users.
        loss_shared_user = torch.tensor(0.0, device=self.device)
        if self.shared_user_nce_weight > 0:
            loss_shared_user = self._shared_user_nce_loss(
                s1, s2, inter_d1, inter_d2,
            )

        # Directional MMD alignment between batch users' shared vectors.
        # If ``shared_user_mmd_weight`` is non-zero, A/B both use that value;
        # otherwise ``shared_user_mmd_weight_A/B`` control the two directions.
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

        # Paired overlap-user cosine alignment: a soft pull between the two
        # domains' shared vectors for the same overlap identity, without
        # adding noisy negatives.
        loss_overlap_cos = torch.tensor(0.0, device=self.device)
        if self.shared_overlap_cos_weight > 0:
            loss_overlap_cos = self._overlap_cos_loss(s1, s2, inter_d1, inter_d2)

        # Shared-channel ranking distillation: the shared vector matches the
        # fused user's detached pos-vs-neg margin instead of taking a direct
        # auxiliary BPR target.
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

        # Shared-channel cross-domain BPR. Valid d1/s1 and d2/s2 terms are
        # averaged, then scaled once by ``shared_cross_bpr_weight``.
        loss_shared_cross_bpr = torch.tensor(0.0, device=self.device)
        if self.shared_cross_bpr_weight > 0:
            d1_cross_terms, d2_cross_terms = self._shared_cross_bpr_loss_terms(
                s1, s2, inter_d1, neg_d1_item, inter_d2, neg_d2_item,
                d1_item_emb, d2_item_emb,
            )
            cross_terms = d1_cross_terms + d2_cross_terms
            if cross_terms:
                loss_shared_cross_bpr = sum(cross_terms) / len(cross_terms)

        # Original standalone shared-channel BPR: directly supervise the
        # shared user vector with the domain's own sampled positive/negative.
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
        """Full in-domain ranking with user-history masking."""
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
