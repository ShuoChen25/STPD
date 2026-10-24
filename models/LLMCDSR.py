import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_pca_item_initializers


class _PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class _ContrastiveLoss2(nn.Module):
    def __init__(self, tau=1):
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax((X_similarity + Y_similarity) / 2 * self.temperature, dim=-1)
        X_loss = self.cross_entropy(logits, targets, reduction="none")
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction="none")
        loss = (Y_loss + X_loss) / 2.0
        return loss.mean()

    @staticmethod
    def cross_entropy(preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"unknown reduction: {reduction}")


class _SASRecBackbone(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.dev = device
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(args.hidden_size, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.forward_layers.append(_PointWiseFeedForward(args.hidden_size, args.dropout_rate))

    def forward(self, seqs, log_seqs):
        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, need_weights=False)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)


class _BaseSeqModel(nn.Module):
    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.freeze_modules = []
        self.filter_init_modules = []

    def _init_weights(self):
        for name, param in self.named_parameters():
            try:
                should_init = True
                for fm in self.filter_init_modules:
                    if fm in name:
                        should_init = False
                if should_init:
                    nn.init.xavier_normal_(param.data)
            except Exception:
                pass


class _LLMArgs:
    def __init__(self, params):
        self.hidden_size = int(params.get("llmcdsr_hidden_size", params.get("embedding_size", 128)))
        self.trm_num = int(params.get("llmcdsr_trm_num", 2))
        self.num_heads = int(params.get("llmcdsr_num_heads", 1))
        self.dropout_rate = float(params.get("llmcdsr_dropout_rate", 0.5))
        self.max_len = int(params.get("llmcdsr_max_len", 200))
        self.local_emb = bool(params.get("llmcdsr_local_emb", True))
        self.global_emb = bool(params.get("llmcdsr_global_emb", True))
        self.freeze_emb = bool(params.get("llmcdsr_freeze_emb", True))
        self.alpha = float(params.get("llmcdsr_alpha", 0.1))
        self.beta = float(params.get("llmcdsr_beta", 0.1))
        self.tau = float(params.get("llmcdsr_tau", 1.0))
        self.tau_reg = float(params.get("llmcdsr_tau_reg", 1.0))
        self.train_chunk_size = int(params.get("llmcdsr_train_chunk_size", 256))
        self.eval_user_chunk_size = int(
            params.get("llmcdsr_eval_user_chunk_size", self.train_chunk_size)
        )
        self.eval_item_chunk_size = int(params.get("llmcdsr_eval_item_chunk_size", 8192))
        self.cpu_state_dict = bool(params.get("llmcdsr_cpu_state_dict", True))


def _pad_pretrained_items(items, dtype):
    return torch.cat([torch.zeros(1, items.shape[1], dtype=dtype), items.to(dtype=dtype)], dim=0)


def _is_cuda_oom(exc):
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in message and "out of memory" in message
    )


def _resolve_user_embedding_path(args):
    pair = f"{args.d1['name'].lower()}_{args.d2['name'].lower()}"
    base = os.path.join("LLM", "embedding_output")
    candidates = [
        os.path.join(base, f"{pair}_user_summaries_embeddings_sorted.json"),
        os.path.join(base, f"{pair}_user_summaries_embeddings.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def _vector_from_user_record(record):
    if isinstance(record, dict):
        for key in ("SharedInterest", "summary", "embedding"):
            vec = record.get(key)
            if isinstance(vec, list):
                return vec
    if isinstance(record, list):
        return record
    return None


def _load_user_summary_embeddings(args, hidden_size, dtype):
    source_path = _resolve_user_embedding_path(args)
    cache_path = os.path.join(args.material_path, "user_summary_init.pt")

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        return cache["weight"].to(dtype=dtype), cache["valid_mask"]

    if not os.path.exists(source_path):
        weight = torch.zeros(1, hidden_size, dtype=dtype)
        valid_mask = torch.zeros(1, dtype=torch.bool)
        return weight, valid_mask

    with open(source_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    rows = []
    max_user = -1
    raw_dim = None
    for user_id_str, record in payload.items():
        try:
            user_id = int(user_id_str)
        except (TypeError, ValueError):
            continue
        vec = _vector_from_user_record(record)
        if not isinstance(vec, list):
            continue
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            continue
        if raw_dim is None:
            raw_dim = int(arr.size)
        if arr.size != raw_dim:
            continue
        rows.append((user_id, arr))
        max_user = max(max_user, user_id)

    if not rows:
        weight = torch.zeros(1, hidden_size, dtype=dtype)
        valid_mask = torch.zeros(1, dtype=torch.bool)
        return weight, valid_mask

    n_rows = max(max_user + 1, int(args.data.get("n_shared_users", 0)))
    weight = torch.zeros(n_rows, raw_dim, dtype=torch.float32)
    valid_mask = torch.zeros(n_rows, dtype=torch.bool)
    for user_id, arr in rows:
        if 0 <= user_id < n_rows:
            weight[user_id] = torch.from_numpy(arr)
            valid_mask[user_id] = True

    torch.save({"weight": weight, "valid_mask": valid_mask}, cache_path)
    return weight.to(dtype=dtype), valid_mask


class LLMCDSR_base(_BaseSeqModel):
    disable_compile = True

    def __init__(self, args):
        params = args.params
        self.cfg = _LLMArgs(params)
        self.item_numA = int(args.d1["n_items"])
        self.item_numB = int(args.d2["n_items"])
        item_num = self.item_numA + self.item_numB

        super().__init__(max(args.d1["n_users"], args.d2["n_users"]), item_num, args.device, self.cfg)

        self.device = args.device
        self.dtype = args.torch_type
        self.max_len = self.cfg.max_len
        self.train_chunk_size = self.cfg.train_chunk_size
        self.eval_user_chunk_size = self.cfg.eval_user_chunk_size
        self.eval_item_chunk_size = self.cfg.eval_item_chunk_size
        self.cpu_state_dict = self.cfg.cpu_state_dict
        self.global_emb = self.cfg.global_emb
        self.d1_n_users = int(args.d1["n_users"])
        self.d2_n_users = int(args.d2["n_users"])
        self.n_shared_users = int(args.data["n_shared_users"])
        self.d1_domain_name = args.d1.get("name", "domain1").lower()
        self.d2_domain_name = args.d2.get("name", "domain2").lower()

        item_init = load_pca_item_initializers(args, self.cfg.hidden_size, dtype=self.dtype)
        has_item_init = item_init is not None

        if has_item_init:
            d1_item_init = item_init["d1_item_init"].to(dtype=self.dtype)
            d2_item_init = item_init["d2_item_init"].to(dtype=self.dtype)
            llm_item_emb = torch.cat(
                [
                    torch.zeros(1, self.cfg.hidden_size, dtype=self.dtype),
                    d1_item_init,
                    d2_item_init,
                ],
                dim=0,
            )
        else:
            d1_item_init = d2_item_init = None
            llm_item_emb = None

        if self.cfg.global_emb and has_item_init:
            self.item_emb_llm = nn.Embedding.from_pretrained(
                llm_item_emb, freeze=self.cfg.freeze_emb, padding_idx=0
            )
        else:
            self.item_emb_llm = nn.Embedding(item_num + 1, self.cfg.hidden_size, padding_idx=0)

        adapter_mid = max(1, self.item_emb_llm.embedding_dim // 2)
        self.adapter = nn.Sequential(
            nn.Linear(self.item_emb_llm.embedding_dim, adapter_mid),
            nn.Linear(adapter_mid, self.cfg.hidden_size),
        )

        self.pos_emb = nn.Embedding(self.cfg.max_len + 1, self.cfg.hidden_size)
        self.emb_dropout = nn.Dropout(p=self.cfg.dropout_rate)
        self.backbone = _SASRecBackbone(args.device, self.cfg)

        if self.cfg.local_emb and has_item_init:
            self.item_embA = nn.Embedding.from_pretrained(
                _pad_pretrained_items(d1_item_init, self.dtype), freeze=True, padding_idx=0
            )
        else:
            self.item_embA = nn.Embedding(self.item_numA + 1, self.cfg.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(self.cfg.max_len + 1, self.cfg.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=self.cfg.dropout_rate)
        self.backboneA = _SASRecBackbone(args.device, self.cfg)

        if self.cfg.local_emb and has_item_init:
            self.item_embB = nn.Embedding.from_pretrained(
                _pad_pretrained_items(d2_item_init, self.dtype), freeze=True, padding_idx=0
            )
        else:
            self.item_embB = nn.Embedding(self.item_numB + 1, self.cfg.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(self.cfg.max_len + 1, self.cfg.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=self.cfg.dropout_rate)
        self.backboneB = _SASRecBackbone(args.device, self.cfg)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        if self.cfg.global_emb and has_item_init:
            self.filter_init_modules.append("item_emb_llm")
        if self.cfg.local_emb and has_item_init:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")

        self._init_weights()

        self.d1_train_hist = self._build_history(args.d1["train"], self.d1_n_users)
        self.d2_train_hist = self._build_history(args.d2["train"], self.d2_n_users)
        self.d1_train_valid_hist = self._build_history(
            args.d1["train"] + args.d1["valid"], self.d1_n_users
        )
        self.d2_train_valid_hist = self._build_history(
            args.d2["train"] + args.d2["valid"], self.d2_n_users
        )

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if not self.cpu_state_dict:
            return state
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()
        return state

    @staticmethod
    def _build_history(rows, n_users):
        histories = [[] for _ in range(n_users)]
        for row in rows:
            user, item = int(row[0]), int(row[1])
            if 0 <= user < n_users:
                histories[user].append(item + 1)
        return histories

    @staticmethod
    def _without_item(history, item):
        return [x for x in history if x != item]

    def _get_embedding(self, log_seqs, domain="A"):
        if domain == "A":
            item_seq_emb = self.item_embA(log_seqs)
        elif domain == "B":
            item_seq_emb = self.item_embB(log_seqs)
        elif domain == "AB":
            if self.global_emb:
                item_seq_emb = self.item_emb_llm(log_seqs)
                item_seq_emb = self.adapter(item_seq_emb)
            else:
                item_seq_emb = self.item_emb_llm(log_seqs)
        else:
            raise ValueError
        return item_seq_emb

    def log2feats(self, log_seqs, positions, domain="A"):
        if domain == "AB":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            log_feats = self.backbone(seqs, log_seqs)
        elif domain == "A":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)
            log_feats = self.backboneA(seqs, log_seqs)
        elif domain == "B":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)
            log_feats = self.backboneB(seqs, log_seqs)
        else:
            raise ValueError
        return log_feats

    def _bce_terms(self, pos_logits, neg_logits, pos):
        indices = pos != 0
        if not torch.any(indices):
            return pos_logits.sum() * 0.0
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        return (pos_loss + neg_loss).mean()

    def forward(
        self,
        seq,
        pos,
        neg,
        positions,
        seqA,
        posA,
        negA,
        positionsA,
        seqB,
        posB,
        negB,
        positionsB,
        target_domain,
        domain_mask,
        **kwargs,
    ):
        log_feats = self.log2feats(seq, positions, domain="AB")
        pos_embs = self._get_embedding(pos, domain="AB")
        neg_embs = self._get_embedding(neg, domain="AB")
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        log_featsA = self.log2feats(seqA, positionsA, domain="A")
        pos_embsA = self._get_embedding(posA, domain="A")
        neg_embsA = self._get_embedding(negA, domain="A")
        pos_logitsA = (log_featsA * pos_embsA).sum(dim=-1)
        neg_logitsA = (log_featsA * neg_embsA).sum(dim=-1)
        if torch.any(posA > 0):
            pos_logitsA[posA > 0] += pos_logits[domain_mask == 0]
            neg_logitsA[posA > 0] += neg_logits[domain_mask == 0]

        log_featsB = self.log2feats(seqB, positionsB, domain="B")
        pos_embsB = self._get_embedding(posB, domain="B")
        neg_embsB = self._get_embedding(negB, domain="B")
        pos_logitsB = (log_featsB * pos_embsB).sum(dim=-1)
        neg_logitsB = (log_featsB * neg_embsB).sum(dim=-1)
        if torch.any(posB > 0):
            pos_logitsB[posB > 0] += pos_logits[domain_mask == 1]
            neg_logitsB[posB > 0] += neg_logits[domain_mask == 1]

        return self._bce_terms(pos_logitsA, neg_logitsA, posA) + self._bce_terms(
            pos_logitsB, neg_logitsB, posB
        )

    def _seq_arrays(self, context, target=0, neg=0, target_domain=-1):
        seq = np.zeros(self.max_len, dtype=np.int64)
        pos = np.zeros(self.max_len, dtype=np.int64)
        neg_arr = np.zeros(self.max_len, dtype=np.int64)
        positions = np.zeros(self.max_len, dtype=np.int64)
        domain_mask = np.ones(self.max_len, dtype=np.int64) * -1

        kept = context[-self.max_len :]
        if kept:
            seq[-len(kept) :] = kept
            positions[-len(kept) :] = np.arange(1, len(kept) + 1)
        if target > 0:
            pos[-1] = target
            neg_arr[-1] = neg
            domain_mask[-1] = target_domain
        return seq, pos, neg_arr, positions, domain_mask

    def _reg_pair(self, user, domain, target_local):
        d1_hist = self.d1_train_hist[user] if user < self.d1_n_users else []
        d2_hist = self.d2_train_hist[user] if user < self.n_shared_users else []
        if domain == 1:
            d1_hist = self._without_item(d1_hist, target_local)
        else:
            d2_hist = self._without_item(d2_hist, target_local)
        if not d1_hist or not d2_hist:
            return 0, 0
        return int(np.random.choice(d1_hist)), int(self.item_numA + np.random.choice(d2_hist))

    def _batch_inputs(self, inter, neg_items, domain):
        seq, pos, neg, positions = [], [], [], []
        seqA, posA, negA, positionsA = [], [], [], []
        seqB, posB, negB, positionsB = [], [], [], []
        target_domain, domain_mask, reg_A, reg_B, user_id = [], [], [], [], []

        inter_cpu = inter.detach().cpu().numpy()
        neg_cpu = neg_items.detach().cpu().numpy()
        for row, neg_item in zip(inter_cpu, neg_cpu):
            user = int(row[0])
            target_local = int(row[1]) + 1
            neg_local = int(neg_item) + 1

            d1_hist = self.d1_train_hist[user] if user < self.d1_n_users else []
            d2_hist = self.d2_train_hist[user] if user < self.d2_n_users else []
            if user >= self.n_shared_users:
                if domain == 1:
                    d2_hist = []
                else:
                    d1_hist = []

            if domain == 1:
                cur_d1 = self._without_item(d1_hist, target_local)
                cur_d2 = d2_hist
                ab_context = [self.item_numA + x for x in cur_d2] + cur_d1
                ab_target, ab_neg, t_domain = target_local, neg_local, 0
                a_target, a_neg = target_local, neg_local
                b_target, b_neg = 0, 0
            else:
                cur_d1 = d1_hist
                cur_d2 = self._without_item(d2_hist, target_local)
                ab_context = cur_d1 + [self.item_numA + x for x in cur_d2]
                ab_target, ab_neg, t_domain = self.item_numA + target_local, self.item_numA + neg_local, 1
                a_target, a_neg = 0, 0
                b_target, b_neg = target_local, neg_local

            s, p, n, ps, dm = self._seq_arrays(ab_context, ab_target, ab_neg, t_domain)
            sA, pA, nA, psA, _ = self._seq_arrays(cur_d1, a_target, a_neg, 0)
            sB, pB, nB, psB, _ = self._seq_arrays(cur_d2, b_target, b_neg, 1)
            rA, rB = self._reg_pair(user, domain, target_local)

            seq.append(s)
            pos.append(p)
            neg.append(n)
            positions.append(ps)
            seqA.append(sA)
            posA.append(pA)
            negA.append(nA)
            positionsA.append(psA)
            seqB.append(sB)
            posB.append(pB)
            negB.append(nB)
            positionsB.append(psB)
            target_domain.append(t_domain)
            domain_mask.append(dm)
            reg_A.append([rA])
            reg_B.append([rB])
            user_id.append(user)

        device = inter.device
        return {
            "seq": torch.tensor(np.asarray(seq), dtype=torch.long, device=device),
            "pos": torch.tensor(np.asarray(pos), dtype=torch.long, device=device),
            "neg": torch.tensor(np.asarray(neg), dtype=torch.long, device=device),
            "positions": torch.tensor(np.asarray(positions), dtype=torch.long, device=device),
            "seqA": torch.tensor(np.asarray(seqA), dtype=torch.long, device=device),
            "posA": torch.tensor(np.asarray(posA), dtype=torch.long, device=device),
            "negA": torch.tensor(np.asarray(negA), dtype=torch.long, device=device),
            "positionsA": torch.tensor(np.asarray(positionsA), dtype=torch.long, device=device),
            "seqB": torch.tensor(np.asarray(seqB), dtype=torch.long, device=device),
            "posB": torch.tensor(np.asarray(posB), dtype=torch.long, device=device),
            "negB": torch.tensor(np.asarray(negB), dtype=torch.long, device=device),
            "positionsB": torch.tensor(np.asarray(positionsB), dtype=torch.long, device=device),
            "target_domain": torch.tensor(target_domain, dtype=torch.long, device=device),
            "domain_mask": torch.tensor(np.asarray(domain_mask), dtype=torch.long, device=device),
            "reg_A": torch.tensor(reg_A, dtype=torch.long, device=device),
            "reg_B": torch.tensor(reg_B, dtype=torch.long, device=device),
            "user_id": torch.tensor(user_id, dtype=torch.long, device=device),
        }

    @staticmethod
    def _cat_inputs(chunks):
        keys = chunks[0].keys()
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}

    def _calculate_loss_chunk(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        chunks = []
        if inter_d1.numel() > 0:
            chunks.append(self._batch_inputs(inter_d1, neg_d1_item, domain=1))
        if inter_d2.numel() > 0:
            chunks.append(self._batch_inputs(inter_d2, neg_d2_item, domain=2))
        if not chunks:
            return next(self.parameters()).sum() * 0.0
        inputs = self._cat_inputs(chunks)
        return self.forward(**inputs)

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        return self._calculate_loss_chunk(inter_d1, neg_d1_item, inter_d2, neg_d2_item)

    def _iter_training_chunks(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item, chunk_size):
        total = int(inter_d1.size(0) + inter_d2.size(0))
        if total <= 0:
            return []
        if chunk_size is None or chunk_size <= 0:
            n_chunks = 1
        else:
            n_chunks = max(1, math.ceil(total / chunk_size))

        d1_inter_parts = torch.tensor_split(inter_d1, n_chunks, dim=0)
        d1_neg_parts = torch.tensor_split(neg_d1_item, n_chunks, dim=0)
        d2_inter_parts = torch.tensor_split(inter_d2, n_chunks, dim=0)
        d2_neg_parts = torch.tensor_split(neg_d2_item, n_chunks, dim=0)

        chunks = []
        for parts in zip(d1_inter_parts, d1_neg_parts, d2_inter_parts, d2_neg_parts):
            d1_part, _, d2_part, _ = parts
            if d1_part.numel() > 0 or d2_part.numel() > 0:
                chunks.append(parts)
        return chunks

    def _train_batch_with_chunk_size(
        self, optimizer, inter_d1, neg_d1_item, inter_d2, neg_d2_item, chunk_size
    ):
        optimizer.zero_grad(set_to_none=True)
        chunks = self._iter_training_chunks(
            inter_d1, neg_d1_item, inter_d2, neg_d2_item, chunk_size
        )
        if not chunks:
            return next(self.parameters()).sum() * 0.0

        total_loss = torch.zeros((), device=self.device)
        scale = 1.0 / len(chunks)
        for d1_part, d1_neg_part, d2_part, d2_neg_part in chunks:
            loss = self._calculate_loss_chunk(d1_part, d1_neg_part, d2_part, d2_neg_part)
            (loss * scale).backward()
            total_loss = total_loss + loss.detach()

        optimizer.step()
        return total_loss * scale

    def train_batch(self, optimizer, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        total = int(inter_d1.size(0) + inter_d2.size(0))
        chunk_size = self.train_chunk_size if self.train_chunk_size > 0 else max(total, 1)

        while True:
            try:
                loss = self._train_batch_with_chunk_size(
                    optimizer, inter_d1, neg_d1_item, inter_d2, neg_d2_item, chunk_size
                )
                self.train_chunk_size = chunk_size
                return loss
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or chunk_size <= 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)

    def _eval_inputs(self, users, domain, mode):
        d1_histories = self.d1_train_hist if mode == "valid" else self.d1_train_valid_hist
        d2_histories = self.d2_train_hist if mode == "valid" else self.d2_train_valid_hist
        seq, positions, seqA, positionsA, seqB, positionsB = [], [], [], [], [], []

        for user in users.detach().cpu().tolist():
            d1_hist = d1_histories[user] if user < self.d1_n_users else []
            d2_hist = d2_histories[user] if user < self.d2_n_users else []
            if user >= self.n_shared_users:
                if domain == 1:
                    d2_hist = []
                else:
                    d1_hist = []

            if domain == 1:
                ab_context = [self.item_numA + x for x in d2_hist] + d1_hist
            else:
                ab_context = d1_hist + [self.item_numA + x for x in d2_hist]

            s, _, _, ps, _ = self._seq_arrays(ab_context)
            sA, _, _, psA, _ = self._seq_arrays(d1_hist)
            sB, _, _, psB, _ = self._seq_arrays(d2_hist)
            seq.append(s)
            positions.append(ps)
            seqA.append(sA)
            positionsA.append(psA)
            seqB.append(sB)
            positionsB.append(psB)

        device = users.device
        return (
            torch.tensor(np.asarray(seq), dtype=torch.long, device=device),
            torch.tensor(np.asarray(positions), dtype=torch.long, device=device),
            torch.tensor(np.asarray(seqA), dtype=torch.long, device=device),
            torch.tensor(np.asarray(positionsA), dtype=torch.long, device=device),
            torch.tensor(np.asarray(seqB), dtype=torch.long, device=device),
            torch.tensor(np.asarray(positionsB), dtype=torch.long, device=device),
        )

    def _candidate_embeddings(self, domain):
        if domain == 1:
            ab_ids = torch.arange(1, self.item_numA + 1, device=self.device)
            local_ids = torch.arange(1, self.item_numA + 1, device=self.device)
            return self._get_embedding(ab_ids, domain="AB"), self._get_embedding(local_ids, domain="A")

        ab_ids = torch.arange(
            self.item_numA + 1, self.item_numA + self.item_numB + 1, device=self.device
        )
        local_ids = torch.arange(1, self.item_numB + 1, device=self.device)
        return self._get_embedding(ab_ids, domain="AB"), self._get_embedding(local_ids, domain="B")

    def _eval_user_features(self, users, domain, mode):
        seq, positions, seqA, positionsA, seqB, positionsB = self._eval_inputs(users, domain, mode)
        final_feat = self.log2feats(seq, positions, domain="AB")[:, -1, :]
        if domain == 1:
            final_local = self.log2feats(seqA, positionsA, domain="A")[:, -1, :]
        else:
            final_local = self.log2feats(seqB, positionsB, domain="B")[:, -1, :]
        return final_feat, final_local

    @torch.no_grad()
    def _eval_domain(self, eval_set, domain, mode, history, domain_name):
        item_ab, item_local = self._candidate_embeddings(domain)
        n_items = item_ab.size(0)
        user_step = self.eval_user_chunk_size
        item_step = self.eval_item_chunk_size
        metrics = {"Hit@10": 0, "NDCG@10": 0, "Hit@20": 0, "NDCG@20": 0}
        total = len(eval_set.dataset)

        for batch in eval_set:
            rows = batch[0]
            users_all = rows[:, 0].long()
            pos_all = rows[:, 1].long()
            cur_user_step = user_step if user_step > 0 else users_all.size(0)
            cur_item_step = item_step if item_step > 0 else n_items

            for u_start in range(0, users_all.size(0), cur_user_step):
                u_end = min(u_start + cur_user_step, users_all.size(0))
                users = users_all[u_start:u_end]
                pos = pos_all[u_start:u_end]
                final_feat, final_local = self._eval_user_features(users, domain, mode)
                pos_scores = (
                    (final_feat * item_ab[pos]).sum(dim=1)
                    + (final_local * item_local[pos]).sum(dim=1)
                )

                ranks = torch.ones_like(pos, dtype=torch.long)
                user_list = users.tolist()
                for item_start in range(0, n_items, cur_item_step):
                    item_end = min(item_start + cur_item_step, n_items)
                    scores = (
                        final_feat @ item_ab[item_start:item_end].T
                        + final_local @ item_local[item_start:item_end].T
                    )

                    if history is not None:
                        for row_idx, user in enumerate(user_list):
                            hist_items = history[user]
                            if hist_items.size == 0:
                                continue
                            in_chunk = (hist_items >= item_start) & (hist_items < item_end)
                            if np.any(in_chunk):
                                scores[row_idx, hist_items[in_chunk] - item_start] = float("-inf")

                    ranks = ranks + (scores > pos_scores.unsqueeze(1)).sum(dim=1)

                ranks_f = ranks.float()
                for cutoff in (10, 20):
                    hit = ranks <= cutoff
                    metrics[f"Hit@{cutoff}"] += hit.sum().item()
                    metrics[f"NDCG@{cutoff}"] += (
                        1.0 / torch.log2(ranks_f[hit] + 1)
                    ).sum().item()

        metrics = {name: value / total for name, value in metrics.items()}
        print(f"{domain_name}")
        print(
            f"Hit@10: {metrics['Hit@10']:.4f}, NDCG@10: {metrics['NDCG@10']:.4f}, "
            f"Hit@20: {metrics['Hit@20']:.4f}, NDCG@20: {metrics['NDCG@20']:.4f}"
        )
        return {"domain": domain_name, **metrics}

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=None):
        if domain == 1:
            return self._eval_domain(eval_set_1, 1, mode, mask_d1, self.d1_domain_name)
        if domain == 2:
            return self._eval_domain(eval_set_2, 2, mode, mask_d2, self.d2_domain_name)
        d1 = self._eval_domain(eval_set_1, 1, mode, mask_d1, self.d1_domain_name)
        d2 = self._eval_domain(eval_set_2, 2, mode, mask_d2, self.d2_domain_name)
        return [d1, d2]


class LLMCDSR(LLMCDSR_base):
    def __init__(self, args):
        super().__init__(args)

        self.alpha = self.cfg.alpha
        self.beta = self.cfg.beta

        user_weight, user_valid = _load_user_summary_embeddings(args, self.cfg.hidden_size, self.dtype)
        self.user_emb_llm = nn.Embedding.from_pretrained(user_weight, freeze=True, padding_idx=0)
        self.register_buffer("user_emb_valid", user_valid, persistent=False)

        user_mid = max(1, self.user_emb_llm.embedding_dim // 2)
        self.user_adapter = nn.Sequential(
            nn.Linear(self.user_emb_llm.embedding_dim, user_mid),
            nn.Linear(user_mid, self.cfg.hidden_size),
        )

        self.reg_loss_func = _ContrastiveLoss2(tau=self.cfg.tau_reg)
        self.user_loss_func = _ContrastiveLoss2(tau=self.cfg.tau)

        self.filter_init_modules.append("user_emb_llm")
        self._init_weights()

    def _valid_user_mask(self, user_id):
        clamped = user_id.clamp(max=self.user_emb_llm.num_embeddings - 1)
        in_range = user_id < self.user_emb_llm.num_embeddings
        return clamped, in_range & self.user_emb_valid[clamped]

    def forward(
        self,
        seq,
        pos,
        neg,
        positions,
        seqA,
        posA,
        negA,
        positionsA,
        seqB,
        posB,
        negB,
        positionsB,
        target_domain,
        domain_mask,
        reg_A,
        reg_B,
        user_id,
        **kwargs,
    ):
        loss = super().forward(
            seq,
            pos,
            neg,
            positions,
            seqA,
            posA,
            negA,
            positionsA,
            seqB,
            posB,
            negB,
            positionsB,
            target_domain,
            domain_mask,
            **kwargs,
        )

        if self.alpha > 0:
            reg_A = reg_A.view(-1)
            reg_B = reg_B.view(-1)
            reg_mask = (reg_A > 0) & (reg_B > 0)
            if torch.any(reg_mask):
                reg_A_emb = self._get_embedding(reg_A[reg_mask], domain="AB")
                reg_B_emb = self._get_embedding(reg_B[reg_mask], domain="AB")
                loss = loss + self.alpha * self.reg_loss_func(reg_A_emb, reg_B_emb)

        if self.beta > 0:
            user_id = user_id.long()
            user_lookup, valid_user = self._valid_user_mask(user_id)
            if torch.any(valid_user):
                log_feats = self.log2feats(seq, positions, domain="AB")
                final_feat = log_feats[:, -1, :][valid_user]
                llm_feats = self.user_emb_llm(user_lookup[valid_user])
                llm_feats = self.user_adapter(llm_feats)
                loss = loss + self.beta * self.user_loss_func(llm_feats, final_feat)

        return loss


LLM4CDSR = LLMCDSR
