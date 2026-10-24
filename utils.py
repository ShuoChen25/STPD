import os
import json
import math
import yaml
import torch
import scipy.sparse as sp
import numpy as np
import random
from collections import defaultdict


def convert_sp_mat_to_sp_tensor(sp_mat, dtype=torch.float32):
    coo = sp_mat.tocoo()
    indices = torch.LongTensor(np.array([coo.row, coo.col]))
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()


def symmetric_normalize(sp_mat):
    rowsum = np.array(sp_mat.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    np.power(rowsum, -0.5, out=d_inv_sqrt, where=rowsum > 0)
    D = sp.diags(d_inv_sqrt)
    return D.dot(sp_mat).dot(D).tocsr()



def load_tri(path, mode):
    with open(os.path.join(path, f"{mode}.txt"), "r") as file:
        data = file.readlines()[1:]
        data = [tri.strip('\n').split('\t') for tri in data]
        data = [list(map(int, tri)) for tri in data]

    return data


def get_inter_mat(args):
    d1_n_users, d2_n_users = args.d1['n_users'], args.d2['n_users']
    n_users = d1_n_users + d2_n_users - args.n_shared_users
    d1_n_items, d2_n_items = args.d1['n_items'], args.d2['n_items']

    check_paths(f"materials/{args.dataset}", f"materials/{args.dataset}/{args.domains}")
    path_root = f"materials/{args.dataset}/{args.domains}"
    
    if os.path.exists(f"{path_root}/d1_inters.npy") and os.path.exists(f"{path_root}/d2_inters.npy"):

        d1_inters = np.load(f"{path_root}/d1_inters.npy")
        row, col, data = d1_inters[:, 0], d1_inters[:, 1], np.ones(len(d1_inters))
        args.d1['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d1_n_items, n_users+d1_n_items))

        d2_inters = np.load(f"{path_root}/d2_inters.npy")
        row, col, data = d2_inters[:, 0], d2_inters[:, 1], np.ones(len(d2_inters))
        args.d2['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d2_n_items, n_users+d2_n_items))

        return

    d1_inters = []
    for inter in args.d1['train']:
        user, item = inter[0], inter[1]
        d1_inters.append((user, item))
    d1_inters = [(u, v + n_users) for (u, v) in d1_inters]
    d1_inters += [(v, u) for (u, v) in d1_inters]

    d1_n_inters = len(d1_inters)

    d1_inters = np.array(d1_inters)
    np.save(f"{path_root}/d1_inters.npy", d1_inters)
    row, col, data = d1_inters[:, 0], d1_inters[:, 1], np.ones(d1_n_inters)
    args.d1['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d1_n_items, n_users+d1_n_items))

    d2_inters = []
    for inter in args.d2['train']:
        user, item = inter[0], inter[1]
        if user < args.n_shared_users:
            d2_inters.append((user, item))
        else:
            d2_inters.append((user+args.d1['n_users']-args.n_shared_users, item))
             
    d2_inters = [(u, v + n_users) for (u, v) in d2_inters]
    d2_inters += [(v, u) for (u, v) in d2_inters]

    d2_n_inters = len(d2_inters)

    d2_inters = np.array(d2_inters)
    np.save(f"{path_root}/d2_inters.npy", d2_inters)
    row, col, data = d2_inters[:, 0], d2_inters[:, 1], np.ones(d2_n_inters)
    args.d2['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d2_n_items, n_users+d2_n_items))


def construct_train(config, user2count, user2inter, neg_ratio):
    data = []
    data += config["train"]
    indices = [i for i in range(config["n_items"])]
    for user in range(config["n_users"]):
        neg = random.sample(indices, user2count[user] * neg_ratio + len(user2inter[user]))
        for item in neg:
            if item not in user2inter[user]:
                data.append([user, item, 0])

    random.shuffle(data)
    data = np.array(data)
    return data


def _sample_easy_neg_column(train_user, user2count, user2inter, n_items):
    total = len(train_user)
    negs = np.empty(total, dtype=np.int64)
    pos = 0
    user = -1
    while pos < total:
        user += 1
        count = user2count[user]
        pos_set = user2inter[user]
        sampled = 0
        while sampled < count:
            batch = np.random.randint(0, n_items, size=(count - sampled) * 2)
            for item in batch:
                if item not in pos_set:
                    negs[pos + sampled] = int(item)
                    sampled += 1
                    if sampled == count:
                        break
        pos += count
    return negs


def construct_neg(train_user, user2count, user2inter, n_items):
    return _sample_easy_neg_column(train_user, user2count, user2inter, n_items)

def construct_inter(config):
    tr = config["train"]
    user2count = defaultdict(int)
    user2item = defaultdict(set)

    for i in range(len(tr)):
        user, item = tr[i][0], tr[i][1]
        user2count[user] += 1
        user2item[user].add(item)

    return user2count, user2item



def build_user_history(rows, n_users):
    buckets = [[] for _ in range(n_users)]
    for row in rows:
        u = int(row[0])
        if 0 <= u < n_users:
            buckets[u].append(int(row[1]))
    return [np.asarray(items, dtype=np.int64) for items in buckets]


def merge_histories(*histories):
    if not histories:
        return []
    n_users = len(histories[0])
    merged = []
    for u in range(n_users):
        seen = set()
        for h in histories:
            arr = h[u]
            if arr.size > 0:
                seen.update(arr.tolist())
        merged.append(np.fromiter(seen, dtype=np.int64))
    return merged


def apply_history_mask(scores, users, histories, fill=float("-inf")):
    if histories is None:
        return scores
    user_list = users.tolist() if torch.is_tensor(users) else list(users)
    for i, u in enumerate(user_list):
        items = histories[u]
        if items.size > 0:
            scores[i, items] = fill
    return scores


def accumulate_full_rank_metrics(scores, pos_items, history, cutoffs=(10, 20)):
    pos_items = pos_items.long()
    pos_scores = scores.gather(1, pos_items.unsqueeze(1))
    if history is not None:
        users_list = list(range(scores.size(0)))
    apply_history_mask(scores, _RangeView(scores.size(0)), None)
    scores.scatter_(1, pos_items.unsqueeze(1), pos_scores)
    ranks = (scores > pos_scores).sum(dim=1) + 1
    ranks_f = ranks.float()
    out = {}
    for cutoff in cutoffs:
        hit = ranks <= cutoff
        out[f"Hit@{cutoff}"] = hit.sum().item()
        out[f"NDCG@{cutoff}"] = (1.0 / torch.log2(ranks_f[hit] + 1)).sum().item()
    return out


class _RangeView:

    def __init__(self, n):
        self._n = n

    def tolist(self):
        return list(range(self._n))


@torch.no_grad()
def full_rank_eval(eval_loader, score_fn, history, domain_name, cutoffs=(10, 20)):
    metrics = {f"Hit@{k}": 0 for k in cutoffs}
    metrics.update({f"NDCG@{k}": 0 for k in cutoffs})
    total = len(eval_loader.dataset)

    for batch in eval_loader:
        rows = batch[0]
        users = rows[:, 0].long()
        pos = rows[:, 1].long()
        scores = score_fn(users)
        pos_scores = scores.gather(1, pos.unsqueeze(1))
        if history is not None:
            for i, u in enumerate(users.tolist()):
                items = history[u]
                if items.size > 0:
                    scores[i, items] = float("-inf")
        scores.scatter_(1, pos.unsqueeze(1), pos_scores)
        ranks = (scores > pos_scores).sum(dim=1) + 1
        ranks_f = ranks.float()
        for cutoff in cutoffs:
            hit = ranks <= cutoff
            metrics[f"Hit@{cutoff}"] += hit.sum().item()
            metrics[f"NDCG@{cutoff}"] += (1.0 / torch.log2(ranks_f[hit] + 1)).sum().item()

    metrics = {name: value / total for name, value in metrics.items()}
    print(f"{domain_name}")
    print(
        f"Hit@10: {metrics['Hit@10']:.4f}, NDCG@10: {metrics['NDCG@10']:.4f}, "
        f"Hit@20: {metrics['Hit@20']:.4f}, NDCG@20: {metrics['NDCG@20']:.4f}"
    )
    return {"domain": domain_name, **metrics}



def _get_item_embedding_output_path(args):
    return os.path.join(
        "LLM",
        "embedding_output",
        f"{args.d1['name'].lower()}_{args.d2['name'].lower()}_item_title_embeddings.json",
    )


def _get_item_init_cache_path(args, emb_dim):
    return os.path.join(args.material_path, f"item_pca_init_{emb_dim}.pt")


def _resolve_domain_payload(domains_payload, target_name):
    if not isinstance(domains_payload, dict):
        return {}
    if target_name in domains_payload and isinstance(domains_payload[target_name], dict):
        return domains_payload[target_name]

    target_lower = target_name.lower()
    for domain_name, domain_payload in domains_payload.items():
        if isinstance(domain_name, str) and domain_name.lower() == target_lower and isinstance(domain_payload, dict):
            return domain_payload
    return {}


def _project_embeddings_with_pca(vectors, out_dim):
    if vectors.size == 0:
        return np.zeros((0, out_dim), dtype=np.float32)

    vectors = vectors.astype(np.float32, copy=False)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    rank = min(out_dim, centered.shape[0] - 1, centered.shape[1])
    if rank <= 0:
        return np.zeros((centered.shape[0], out_dim), dtype=np.float32)

    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    top_components = eigvecs[:, -rank:][:, ::-1].astype(np.float32, copy=False)
    projected = centered @ top_components

    if rank < out_dim:
        projected = np.pad(projected, ((0, 0), (0, out_dim - rank)), mode="constant")

    return projected.astype(np.float32, copy=False)


def load_pca_item_initializers(
    args, emb_dim, dtype=torch.float32, validate_source=True,
):
    source_path = _get_item_embedding_output_path(args)
    cache_path = _get_item_init_cache_path(args, emb_dim)
    expected_d1_items = args.d1["n_items"]
    expected_d2_items = args.d2["n_items"]

    if os.path.exists(cache_path):
        try:
            cache_payload = torch.load(cache_path, map_location="cpu")
        except (RuntimeError, Exception) as e:
            print(f"Warning: corrupted cache file {cache_path}, deleting and rebuilding. ({e})")
            os.remove(cache_path)
            cache_payload = None
        if cache_payload is not None:
            cache_meta = cache_payload.get("meta", {}) if isinstance(cache_payload, dict) else {}
            cache_valid = (
                isinstance(cache_payload, dict)
                and cache_payload.get("d1_item_init") is not None
                and cache_payload.get("d2_item_init") is not None
                and cache_payload["d1_item_init"].shape == (expected_d1_items, emb_dim)
                and cache_payload["d2_item_init"].shape == (expected_d2_items, emb_dim)
                and cache_payload.get("d1_valid_mask") is not None
                and cache_payload.get("d2_valid_mask") is not None
                and cache_payload["d1_valid_mask"].shape == (expected_d1_items,)
                and cache_payload["d2_valid_mask"].shape == (expected_d2_items,)
                and cache_meta.get("emb_dim") == emb_dim
                and cache_meta.get("d1_name") == args.d1["name"]
                and cache_meta.get("d2_name") == args.d2["name"]
            )

            if cache_valid:
                if not validate_source or not os.path.exists(source_path):
                    return {
                        "d1_item_init": cache_payload["d1_item_init"].to(dtype=dtype),
                        "d2_item_init": cache_payload["d2_item_init"].to(dtype=dtype),
                        "mix_item_init": cache_payload["mix_item_init"].to(dtype=dtype),
                        "d1_valid_mask": cache_payload["d1_valid_mask"],
                        "d2_valid_mask": cache_payload["d2_valid_mask"],
                        "mix_valid_mask": cache_payload["mix_valid_mask"],
                        "stats": cache_meta.get("stats", {}),
                        "source": "cache",
                        "source_path": source_path,
                    }

                source_size = os.path.getsize(source_path)
                source_mtime = os.path.getmtime(source_path)
                if (
                    cache_meta.get("source_path") == os.path.abspath(source_path)
                    and cache_meta.get("source_size") == source_size
                    and cache_meta.get("source_mtime") == source_mtime
                ):
                    return {
                        "d1_item_init": cache_payload["d1_item_init"].to(dtype=dtype),
                        "d2_item_init": cache_payload["d2_item_init"].to(dtype=dtype),
                        "mix_item_init": cache_payload["mix_item_init"].to(dtype=dtype),
                        "d1_valid_mask": cache_payload["d1_valid_mask"],
                        "d2_valid_mask": cache_payload["d2_valid_mask"],
                        "mix_valid_mask": cache_payload["mix_valid_mask"],
                        "stats": cache_meta.get("stats", {}),
                        "source": "cache",
                        "source_path": source_path,
                    }

    if not os.path.exists(source_path):
        return None

    with open(source_path, "r", encoding="utf-8") as file:
        item_payload = json.load(file)

    metadata = item_payload.get("metadata", {}) if isinstance(item_payload, dict) else {}
    embedding_key = metadata.get("embedding_key", "TitleEmbedding")
    domains_payload = item_payload.get("domains", item_payload if isinstance(item_payload, dict) else {})

    d1_domain_payload = _resolve_domain_payload(domains_payload, args.d1["name"])
    d2_domain_payload = _resolve_domain_payload(domains_payload, args.d2["name"])

    assignments = []
    vectors = []
    stats = {
        "embedding_key": embedding_key,
        "raw_embedding_dim": None,
        "d1_valid_items": 0,
        "d2_valid_items": 0,
    }

    for domain_name, domain_payload, n_items, stat_key in [
        (args.d1["name"], d1_domain_payload, expected_d1_items, "d1_valid_items"),
        (args.d2["name"], d2_domain_payload, expected_d2_items, "d2_valid_items"),
    ]:
        for item_id_str, item_record in domain_payload.items():
            if not isinstance(item_record, dict):
                continue
            try:
                item_id = int(item_id_str)
            except (TypeError, ValueError):
                continue
            if item_id < 0 or item_id >= n_items:
                continue

            raw_vector = item_record.get(embedding_key)
            if not isinstance(raw_vector, list):
                continue

            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                continue

            if stats["raw_embedding_dim"] is None:
                stats["raw_embedding_dim"] = int(vector.size)
            if vector.size != stats["raw_embedding_dim"]:
                continue

            assignments.append((domain_name, item_id))
            vectors.append(vector)
            stats[stat_key] += 1

    if not vectors:
        return None

    projected = _project_embeddings_with_pca(np.stack(vectors, axis=0), emb_dim)

    d1_item_init = torch.zeros(expected_d1_items, emb_dim, dtype=dtype)
    d2_item_init = torch.zeros(expected_d2_items, emb_dim, dtype=dtype)
    d1_valid_mask = torch.zeros(expected_d1_items, dtype=torch.bool)
    d2_valid_mask = torch.zeros(expected_d2_items, dtype=torch.bool)

    for row_idx, (domain_name, item_id) in enumerate(assignments):
        reduced_vector = torch.from_numpy(projected[row_idx]).to(dtype=dtype)
        if domain_name == args.d1["name"]:
            d1_item_init[item_id] = reduced_vector
            d1_valid_mask[item_id] = True
        else:
            d2_item_init[item_id] = reduced_vector
            d2_valid_mask[item_id] = True

    mix_item_init = torch.cat([d1_item_init, d2_item_init], dim=0)
    mix_valid_mask = torch.cat([d1_valid_mask, d2_valid_mask], dim=0)
    stats["mix_valid_items"] = int(mix_valid_mask.sum().item())

    cache_payload = {
        "meta": {
            "emb_dim": emb_dim,
            "d1_name": args.d1["name"],
            "d2_name": args.d2["name"],
            "source_path": os.path.abspath(source_path),
            "source_size": os.path.getsize(source_path),
            "source_mtime": os.path.getmtime(source_path),
            "stats": stats,
        },
        "d1_item_init": d1_item_init.cpu(),
        "d2_item_init": d2_item_init.cpu(),
        "mix_item_init": mix_item_init.cpu(),
        "d1_valid_mask": d1_valid_mask.cpu(),
        "d2_valid_mask": d2_valid_mask.cpu(),
        "mix_valid_mask": mix_valid_mask.cpu(),
    }
    torch.save(cache_payload, cache_path)

    return {
        "d1_item_init": d1_item_init,
        "d2_item_init": d2_item_init,
        "mix_item_init": mix_item_init,
        "d1_valid_mask": d1_valid_mask,
        "d2_valid_mask": d2_valid_mask,
        "mix_valid_mask": mix_valid_mask,
        "stats": stats,
        "source": "file",
        "source_path": source_path,
    }


def save_material(args, obj, path):
    torch.save(obj, os.path.join(args.material_path, path))
    

def load_material(args, path):
    obj = torch.load(os.path.join(args.material_path, path), map_location=args.device)
    return obj


def load_data(args):
    print(f"Dataset: {args.dataset}.  Domains: {args.domains}.")

    args.material_path = os.path.join("./materials", args.dataset, args.domains)
    config_path = os.path.join(args.material_path, "dataset_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    args.n_shared_users = config["n_shared_users"]
    args.d1 = config["domain_1"]
    args.d2 = config["domain_2"]
    
    path_1 = os.path.join(config['path'], config['domain_1']['name'])
    args.d1["train"] = load_tri(path_1, "train")
    args.d1["valid"] = load_tri(path_1, "valid")
    args.d1["test"] = load_tri(path_1, "test")

    path_2 = os.path.join(config['path'], config['domain_2']['name'])
    args.d2["train"] = load_tri(path_2, "train")
    args.d2["valid"] = load_tri(path_2, "valid")
    args.d2["test"] = load_tri(path_2, "test") 

    get_inter_mat(args)

    for d_cfg in (args.d1, args.d2):
        train_hist = build_user_history(d_cfg["train"], d_cfg["n_users"])
        valid_hist = build_user_history(d_cfg["valid"], d_cfg["n_users"])
        d_cfg["valid_mask"] = train_hist
        d_cfg["test_mask"] = merge_histories(train_hist, valid_hist)

    args.data = config   
    args.torch_type = torch.float32



def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    

    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def check_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
