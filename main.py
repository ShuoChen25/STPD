import copy
import argparse
import random
import sys
import os
import time
import yaml
import torch
import inspect
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import *
from models import *
from utils import *
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "t"):
        return True
    if value in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false.")


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', default='Movie-Music', help='Domain pair, e.g. Movie-Music / Cell-Elec / Sport-Cloth.', type=str)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=2026, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--early_stop_patience', default=9, type=int)
    parser.add_argument('--p', default=True, type=str2bool,
                        help='Whether to save user-representation t-SNE plots for supported models.')
    parser.add_argument('--tsne_max_points_per_group', default=5000, type=int,
                        help='Maximum sampled users per group for the user-representation t-SNE plot. '
                             'Use 0 or a negative value to draw all users.')
    parser.add_argument('--tsne_overlap_pairs', default=250, type=int,
                        help='Deprecated; kept for CLI compatibility. Overlap links are not drawn.')
    parser.add_argument('--tsne_perplexity', default=100.0, type=float,
                        help='Requested t-SNE perplexity; clipped automatically below the total point count.')
    parser.add_argument('--tsne_learning_rate', default=10.0, type=float,
                        help='t-SNE learning rate. Use 0 or a negative value for scikit-learn auto.')
    parser.add_argument('--tsne_point_size', default=4.0, type=float,
                        help='Point size for the user-representation t-SNE scatter plot.')
    parser.add_argument('--tsne_point_alpha', default=0.55, type=float,
                        help='Point opacity for the user-representation t-SNE scatter plot.')
    parser.add_argument('--approach', default='STPD', type=str,
                        help='Model to use: AHNCDR / BiTGCF / CDCDR / CDFA / CoNet / CoPD / CrossAug / STPD / wocross / D2TCDR / DiffCDR / DGCDR / DisenCDR / FedPCL_CDR / LightGCN / LLMCDSR / MFGSLAE / VUG')

    args = parser.parse_args()

    args.dataset = "Amazon"

    device = set_device(args.gpu)
    args.device = device

    set_seed(args.seed)

    with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
        config = yaml.safe_load(yaml_file)

    log_dir = 'log/%s/%s' % (args.dataset, args.domains)
    os.makedirs(log_dir, exist_ok=True)
    run_time = time.strftime("%Y%m%d_%H%M%S")
    args.log_file = os.path.join(log_dir, '%s_%s.log' % (args.approach, run_time))

    load_data(args)
    load_models(args, config)

    return args



def load_models(args, config):
    args.params = config
    model = getattr(sys.modules[__name__], args.approach)(args).to(args.device)
    if getattr(model, "disable_compile", False):
        args.model = model
    else:
        try:
            args.model = torch.compile(model, dynamic=True)
        except RuntimeError as e:
            if "Windows not yet supported for torch.compile" not in str(e):
                raise
            print(f"Warning: torch.compile is unavailable on this platform; using eager mode. ({e})")
            args.model = model
    if hasattr(args.model, "configure_optimizers"):
        args.optim = args.model.configure_optimizers(config, args.lr)
    else:
        optim_kwargs = {"lr": args.lr}
        weight_decay = float(config.get("weight_decay", 0) or 0)
        if weight_decay > 0:
            optim_kwargs["weight_decay"] = weight_decay
        args.optim = getattr(sys.modules[__name__], config["optim"])(args.model.parameters(), **optim_kwargs)


def train(args):
    init_metrics = evaluate(args, 0, "valid")

    def early_stop_score(metric):
        return 0.5 * metric["NDCG@10"] + 0.5 * metric["Hit@10"]

    best_d1_score = early_stop_score(init_metrics["domains"][0])
    best_d2_score = early_stop_score(init_metrics["domains"][1])
    best_d1_epoch, best_d2_epoch = 0, 0
    best_state_d1 = copy.deepcopy(args.model.state_dict())
    best_state_d2 = copy.deepcopy(args.model.state_dict())
    best_avg_score = 0.5 * (best_d1_score + best_d2_score)
    best_avg_epoch = 0
    best_state_avg = copy.deepcopy(args.model.state_dict())
    no_improve_d1, no_improve_d2 = 0, 0

    print(
        f"Current Avg Hit@10: {init_metrics['Hit@10']:.4f}, "
        f"Current Avg NDCG@10: {init_metrics['NDCG@10']:.4f}, "
        f"Current Avg Hit@20: {init_metrics['Hit@20']:.4f}, "
        f"Current Avg NDCG@20: {init_metrics['NDCG@20']:.4f}"
    )
    print(
        f"Best d1 Score@10: {best_d1_score:.4f} (Epoch 0) | "
        f"Best d2 Score@10: {best_d2_score:.4f} (Epoch 0)"
    )

    d1_user2count, d1_user2item = construct_inter(args.d1)
    d2_user2count, d2_user2item = construct_inter(args.d2)

    d1_train = np.array(args.d1["train"])
    d1_train = d1_train[np.argsort(d1_train[:, 0])]
    d2_train = np.array(args.d2["train"])
    d2_train = d2_train[np.argsort(d2_train[:, 0])]

    total_len = len(d1_train) + len(d2_train)
    index_list = list(range(total_len))
    batch_size = args.params["batch_size"]
    n_epoch = args.params["epoch"]
    min_train_epochs = int(getattr(args.model, "min_train_epochs", 0) or 0)

    for i in range(n_epoch):
        if hasattr(args.model, '_epoch'):
            args.model._epoch[0] = i
        args.model.train()

        d1_neg = construct_neg(d1_train[:, 0], d1_user2count, d1_user2item, args.d1["n_items"])
        d2_neg = construct_neg(d2_train[:, 0], d2_user2count, d2_user2item, args.d2["n_items"])

        d1_index = np.arange(len(d1_train))
        np.random.shuffle(d1_index)
        d2_index = np.arange(len(d2_train))
        np.random.shuffle(d2_index)

        d1_pos, d1_neg = d1_train[d1_index], d1_neg[d1_index]
        d2_pos, d2_neg = d2_train[d2_index], d2_neg[d2_index]

        d1_point, d2_point = 0, 0
        random.shuffle(index_list)

        progress_bar = tqdm(
            range(0, total_len, batch_size),
            desc=f"TRAIN - Epoch {i+1}",
            total=(total_len + batch_size - 1) // batch_size,
        )

        for batch_start in progress_bar:
            indices = np.array(index_list[batch_start:min(batch_start + batch_size, total_len)])
            d1_num = int(np.sum(indices < len(d1_train)))
            d2_num = int(np.sum(indices >= len(d1_train)))

            d1_inter = torch.from_numpy(d1_pos[d1_point:d1_point + d1_num]).to(args.device)
            d1_neg_item = torch.from_numpy(d1_neg[d1_point:d1_point + d1_num]).to(args.device)
            d2_inter = torch.from_numpy(d2_pos[d2_point:d2_point + d2_num]).to(args.device)
            d2_neg_item = torch.from_numpy(d2_neg[d2_point:d2_point + d2_num]).to(args.device)

            if hasattr(args.model, "train_batch"):
                loss = args.model.train_batch(args.optim, d1_inter, d1_neg_item, d2_inter, d2_neg_item)
            else:
                args.optim.zero_grad()
                loss = args.model.calculate_loss(d1_inter, d1_neg_item, d2_inter, d2_neg_item)
                loss.backward()
                args.optim.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            d1_point += d1_num
            d2_point += d2_num

        if (i+1) % 3 == 0:
            current_metrics = evaluate(args, i+1, "valid")

            d1_score = early_stop_score(current_metrics["domains"][0])
            d2_score = early_stop_score(current_metrics["domains"][1])
            avg_score = 0.5 * (d1_score + d2_score)

            if d1_score >= best_d1_score:
                best_d1_score = d1_score
                best_d1_epoch = i + 1
                best_state_d1 = copy.deepcopy(args.model.state_dict())
                no_improve_d1 = 0
            elif i + 1 >= min_train_epochs:
                no_improve_d1 += 1

            if d2_score >= best_d2_score:
                best_d2_score = d2_score
                best_d2_epoch = i + 1
                best_state_d2 = copy.deepcopy(args.model.state_dict())
                no_improve_d2 = 0
            elif i + 1 >= min_train_epochs:
                no_improve_d2 += 1

            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                best_avg_epoch = i + 1
                best_state_avg = copy.deepcopy(args.model.state_dict())

            print(
                f"Current Avg Hit@10: {current_metrics['Hit@10']:.4f}, "
                f"Current Avg NDCG@10: {current_metrics['NDCG@10']:.4f}, "
                f"Current Avg Hit@20: {current_metrics['Hit@20']:.4f}, "
                f"Current Avg NDCG@20: {current_metrics['NDCG@20']:.4f}"
            )
            print(
                f"Best d1 Score@10: {best_d1_score:.4f} (Epoch {best_d1_epoch}) | "
                f"Best d2 Score@10: {best_d2_score:.4f} (Epoch {best_d2_epoch})"
            )
            if args.early_stop_patience > 0:
                print(f"Early Stop Counter: d1={no_improve_d1}/{args.early_stop_patience}, d2={no_improve_d2}/{args.early_stop_patience}")
                if i + 1 < min_train_epochs:
                    print(f"Early stopping is disabled until epoch {min_train_epochs}.")
                elif no_improve_d1 >= args.early_stop_patience and no_improve_d2 >= args.early_stop_patience:
                    print(f"Early stopping triggered at epoch {i+1}.")
                    break

    print(f"Best d1 from epoch {best_d1_epoch}, best d2 from epoch {best_d2_epoch}.")
    args.best_avg_state = best_state_avg
    args.best_avg_epoch = best_avg_epoch
    args.best_avg_score = best_avg_score
    print("Training process finished!")
    return best_state_d1, best_state_d2


def _sample_user_ids(n_users, n_shared, max_points, rng, overlap_ids=None):
    if n_users <= 0:
        return np.empty(0, dtype=np.int64)

    max_points = int(max_points)
    if max_points <= 0 or max_points >= int(n_users):
        ids = np.arange(int(n_users), dtype=np.int64)
        rng.shuffle(ids)
        return ids

    return rng.choice(int(n_users), size=max_points, replace=False).astype(np.int64)


def _tsne_project(x, seed, PCA, TSNE, requested_perplexity=100.0,
                  requested_learning_rate=10.0):
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32), 0.0, requested_learning_rate

    if x.shape[0] > 2 and x.shape[1] > 2:
        n_components = min(50, x.shape[1], x.shape[0] - 1)
        x_reduced = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    else:
        x_reduced = x

    n_points = int(x_reduced.shape[0])
    max_valid_perplexity = max(1.0, n_points - 1.0)
    requested_perplexity = float(requested_perplexity)
    if requested_perplexity <= 0:
        requested_perplexity = min(100.0, max(30.0, (n_points - 1.0) / 3.0))
    perplexity = min(requested_perplexity, max_valid_perplexity)

    if requested_learning_rate is None or float(requested_learning_rate) <= 0:
        learning_rate = "auto"
    else:
        learning_rate = float(requested_learning_rate)

    tsne_kwargs = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": learning_rate,
        "perplexity": perplexity,
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    return TSNE(**tsne_kwargs).fit_transform(x_reduced), perplexity, learning_rate


def _draw_user_tsne_panel(ax, coords, labels, tensors,
                          point_size, point_alpha):
    group_names = [t[0] for t in tensors]
    group_colors = {t[0]: t[4] for t in tensors}
    group_markers = {t[0]: t[3] for t in tensors}

    for group in group_names:
        mask = labels == group
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=group_colors[group],
            marker=group_markers[group],
            alpha=point_alpha,
            linewidths=0,
            label=group,
            zorder=2,
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")


def _crossaug_shared_total(args, eager_model, d1_user, d2_user):
    shared_dim = int(getattr(eager_model, "shared_dim", args.params.get("shared_dim", 0)))
    n_layers = int(getattr(eager_model, "n_layers", args.params.get("n_layers", 0)))
    shared_total = shared_dim * (n_layers + 1)
    if shared_total <= 0:
        print("[t-SNE] Skip visualization because CrossAug shared_dim/n_layers are invalid.")
        return None
    if shared_total >= d1_user.shape[1] or shared_total >= d2_user.shape[1]:
        print(
            "[t-SNE] Skip visualization because CrossAug shared/private split is invalid: "
            f"shared_total={shared_total}, d1_dim={d1_user.shape[1]}, "
            f"d2_dim={d2_user.shape[1]}."
        )
        return None
    return shared_total


def _extract_user_tsne_tensors(args, eager_model, outputs):
    if args.approach in ("STPD", "wocross"):
        if len(outputs) < 8:
            print("[t-SNE] Skip visualization because forward() did not return s1/s2/p1/p2.")
            return None
        _, _, _, _, s1, s2, p1, p2 = outputs[:8]
        return [
            (args.d1["name"] + " shared", s1, 1, "shared", "o", "#E41A1C"),
            (args.d2["name"] + " shared", s2, 2, "shared", "o", "#377EB8"),
            (args.d1["name"] + " private", p1, 1, "private", "^", "#4DAF4A"),
            (args.d2["name"] + " private", p2, 2, "private", "^", "#984EA3"),
        ]

    if args.approach == "CrossAug":
        if len(outputs) < 4:
            print("[t-SNE] Skip visualization because CrossAug forward() did not return user/item tensors.")
            return None
        d1_user, _, d2_user, _ = outputs[:4]
        shared_total = _crossaug_shared_total(args, eager_model, d1_user, d2_user)
        if shared_total is None:
            return None
        return [
            (args.d1["name"] + " shared", d1_user[:, :shared_total], 1, "shared", "o", "#E41A1C"),
            (args.d2["name"] + " shared", d2_user[:, :shared_total], 2, "shared", "o", "#377EB8"),
            (args.d1["name"] + " private", d1_user[:, shared_total:], 1, "private", "^", "#4DAF4A"),
            (args.d2["name"] + " private", d2_user[:, shared_total:], 2, "private", "^", "#984EA3"),
        ]

    return None


def _tsne_filename_params(args):
    if args.approach == "CrossAug":
        names = ("shared_dim", "n_layers", "n_hh", "al", "ll", "cdl")
    else:
        names = (
            "lna_weight",
            "ortho_weight",
            "shared_domain_adv_weight",
            "private_domain_cls_weight",
            "shared_user_mmd_weight",
            "shared_cross_bpr_weight",
        )
    return "_".join(f"{name}={args.params.get(name, 0)}" for name in names)


def save_user_tsne(args, state_dict=None, tag="best_avg"):
    if args.approach not in ("STPD", "wocross", "CrossAug"):
        return None

    try:
        import tempfile

        mpl_cache_dir = os.path.join(tempfile.gettempdir(), "stpd_matplotlib")
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        print(f"[t-SNE] Skip visualization because a plotting dependency is missing: {exc}")
        return None

    model = args.model
    eager_model = getattr(model, "_orig_mod", model)
    if not hasattr(eager_model, "forward"):
        print("[t-SNE] Skip visualization because the current model has no forward method.")
        return None

    current_state = copy.deepcopy(model.state_dict()) if state_dict is not None else None
    was_training = model.training

    try:
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            outputs = eager_model.forward()
    finally:
        if current_state is not None:
            model.load_state_dict(current_state)
        if was_training:
            model.train()

    tensor_specs = _extract_user_tsne_tensors(args, eager_model, outputs)
    if tensor_specs is None:
        return None

    sample_seed = time.time_ns() % (2 ** 32)
    rng = np.random.default_rng(sample_seed)
    max_points = int(args.tsne_max_points_per_group)
    d1_ids = _sample_user_ids(
        args.d1["n_users"], args.n_shared_users, max_points, rng,
    )
    d2_ids = _sample_user_ids(
        args.d2["n_users"], args.n_shared_users, max_points, rng,
    )

    dims = {int(tensor.shape[1]) for _, tensor, _, _, _, _ in tensor_specs}
    if len(dims) != 1:
        print(
            "[t-SNE] Skip visualization because shared/private vector dimensions differ: "
            + ", ".join(str(dim) for dim in sorted(dims))
        )
        return None

    tensors = []
    blocks = []
    labels = []
    for label, tensor, domain, _, marker, color in tensor_specs:
        ids = d1_ids if domain == 1 else d2_ids
        if ids.size == 0:
            continue
        tensors.append((label, tensor, ids, marker, color))
        block = tensor[torch.as_tensor(ids, device=tensor.device)].detach().float().cpu().numpy()
        blocks.append(block)
        labels.extend([label] * block.shape[0])

    if not blocks:
        print("[t-SNE] Skip visualization because no users were sampled.")
        return None

    x = np.concatenate(blocks, axis=0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_l2 = x / np.maximum(norms, 1e-12)
    coords_l2, actual_perplexity, actual_learning_rate = _tsne_project(
        x_l2, args.seed, PCA, TSNE,
        requested_perplexity=float(args.tsne_perplexity),
        requested_learning_rate=float(args.tsne_learning_rate),
    )
    labels = np.asarray(labels, dtype=object)
    print(
        f"[t-SNE] Drawing user representations: d1_users={d1_ids.size}, "
        f"d2_users={d2_ids.size}, plotted_points={x.shape[0]}, "
        f"sample_seed={sample_seed}, perplexity={actual_perplexity}, "
        f"learning_rate={actual_learning_rate}"
    )

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=320)
    _draw_user_tsne_panel(
        ax, coords_l2, labels, tensors,
        float(args.tsne_point_size), float(args.tsne_point_alpha),
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, legend_labels,
        loc="upper right",
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()

    out_dir = os.path.join("materials", "Pictures")
    os.makedirs(out_dir, exist_ok=True)
    safe_domains = args.domains.replace("-", "_")
    filename_params = _tsne_filename_params(args)
    out_stem = (
        f"{args.dataset}_{safe_domains}_{args.approach}_user_shared_private_tsne_"
        f"{tag}_{filename_params}"
    )
    out_path = os.path.join(out_dir, f"{out_stem}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[t-SNE] Saved {args.approach} user shared/private visualization to {out_path}")
    return out_path


def save_shared_user_tsne(args, state_dict=None, tag="best_avg"):
    return save_user_tsne(args, state_dict, tag)


def save_stpd_prefusion_tsne(args, state_dict=None, tag="best_avg"):
    return save_user_tsne(args, state_dict, tag)


def evaluate(args, epoch=None, mode="test", eval_size=None, domain=None):
    args.model.eval()
    if eval_size is None:
        eval_size = int(args.params.get("eval_batch_size", args.params.get("batch_size", 32)))

    if mode == "valid":
        print(f"VALID - Epoch {epoch}")
    elif domain is None:
        print(f"TEST")

    d1_pairs = torch.tensor([row[:2] for row in args.d1[mode]], dtype=torch.long).to(args.device)
    d2_pairs = torch.tensor([row[:2] for row in args.d2[mode]], dtype=torch.long).to(args.device)
    eval_set_1 = DataLoader(TensorDataset(d1_pairs), batch_size=eval_size)
    eval_set_2 = DataLoader(TensorDataset(d2_pairs), batch_size=eval_size)

    mask_key = "valid_mask" if mode == "valid" else "test_mask"
    mask_d1 = args.d1[mask_key]
    mask_d2 = args.d2[mask_key]

    if domain is not None:
        metrics = args.model.predict(eval_set_1, eval_set_2, mode, mask_d1, mask_d2, domain=domain)
        if mode == "test":
            d_cfg = args.d1 if domain == 1 else args.d2
            with open(args.log_file, "a") as f:
                f.write(f"{d_cfg['name']} Hit@10: {metrics['Hit@10']:.4f}, NDCG@10: {metrics['NDCG@10']:.4f}, "
                        f"Hit@20: {metrics['Hit@20']:.4f}, NDCG@20: {metrics['NDCG@20']:.4f}\n")
        return metrics

    domain_metrics = args.model.predict(eval_set_1, eval_set_2, mode, mask_d1, mask_d2)

    if mode == "test":
        with open(args.log_file, "a") as f:
            f.write("Params:\n")
            f.write(yaml.safe_dump(args.params, sort_keys=False))
            f.write(f"lr: {args.lr}\n")
            f.write(f"{args.d1['name']} Hit@10: {domain_metrics[0]['Hit@10']:.4f}, NDCG@10: {domain_metrics[0]['NDCG@10']:.4f}, "
                    f"Hit@20: {domain_metrics[0]['Hit@20']:.4f}, NDCG@20: {domain_metrics[0]['NDCG@20']:.4f}\n")
            f.write(f"{args.d2['name']} Hit@10: {domain_metrics[1]['Hit@10']:.4f}, NDCG@10: {domain_metrics[1]['NDCG@10']:.4f}, "
                    f"Hit@20: {domain_metrics[1]['Hit@20']:.4f}, NDCG@20: {domain_metrics[1]['NDCG@20']:.4f}\n\n")

    return {
        "domains": domain_metrics,
        "Hit@10": sum(metric["Hit@10"] for metric in domain_metrics) / len(domain_metrics),
        "NDCG@10": sum(metric["NDCG@10"] for metric in domain_metrics) / len(domain_metrics),
        "Hit@20": sum(metric["Hit@20"] for metric in domain_metrics) / len(domain_metrics),
        "NDCG@20": sum(metric["NDCG@20"] for metric in domain_metrics) / len(domain_metrics),
    }

def test(args, best_state_d1, best_state_d2):
    print("TEST")
    with open(args.log_file, "a") as f:
        f.write("Params:\n")
        f.write(yaml.safe_dump(args.params, sort_keys=False))
        f.write(f"lr: {args.lr}\n")
    args.model.load_state_dict(best_state_d1)
    d1_metrics = evaluate(args, domain=1)
    args.model.load_state_dict(best_state_d2)
    d2_metrics = evaluate(args, domain=2)
    with open(args.log_file, "a") as f:
        f.write("\n")
    return d1_metrics, d2_metrics


if __name__ == '__main__':
    args = init()
    best_state_d1, best_state_d2 = train(args)
    if args.p:
        save_user_tsne(
            args,
            getattr(args, "best_avg_state", None),
            tag=f"best_avg_epoch_{getattr(args, 'best_avg_epoch', 0)}",
        )
    test(args, best_state_d1, best_state_d2)
