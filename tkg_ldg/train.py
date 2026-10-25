import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import dgl
import numpy as np
import torch

from .data import HistoryGraphBuilder, augment_inverse, load_dataset
from .metrics import RankingMetrics, answers_for_timestamp, raw_and_filtered
from .model import ModelConfig, TKGLDG

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "ICEWS18": (262, 6),
    "ICEWS05-15": (500, 8),
    "ICEWS14": (365, 3),
    "GDELT": (50, 7),
    "WIKI": (10, 1),
    "YAGO": (50, 1),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def source_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "tkg_ldg").glob("*.py"))
    }


def audit_dataset(root, name, splits, ne, nr):
    result = {"dataset": name, "num_entities": ne, "num_relations": nr, "splits": {}}
    for name_split, snapshots in splits.items():
        path = Path(root) / name / f"{name_split}.txt"
        result["splits"][name_split] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "facts": sum(len(s.triples) for s in snapshots),
            "snapshots": len(snapshots),
            "first_time": snapshots[0].time,
            "last_time": snapshots[-1].time,
        }
    return result


@torch.no_grad()
def evaluate(model, builder, observed, targets, device, batch_size, max_snapshots=None):
    """Predict each snapshot before revealing its facts to subsequent history."""
    model.eval()
    raw, filtered = RankingMetrics(), RankingMetrics()
    history = list(observed)
    selected = targets[:max_snapshots] if max_snapshots else targets
    for snapshot in selected:
        state = model.encode(*builder(history, device))
        triples = augment_inverse(
            torch.as_tensor(snapshot.triples, device=device), model.num_relations
        )

        answers = answers_for_timestamp(triples.cpu())
        for batch in triples.split(batch_size):
            scores = model.scores(state, batch)
            rr, fr = raw_and_filtered(scores, batch, answers)
            raw.add(rr)
            filtered.add(fr)
        history.append(snapshot)
    return {
        "raw": raw.compute(),
        "time_filtered": filtered.compute(),
        "snapshots": len(selected),
        "protocol": "rolling_one_step_ground_truth_history",
    }


def train_epoch(
    model,
    optimizer,
    builder,
    snapshots,
    device,
    batch_size,
    entity_weight,
    grad_clip,
    max_snapshots=None,
    shuffle=True,
):
    model.train()
    indices = list(range(1, len(snapshots)))
    if max_snapshots:

        indices = indices[:max_snapshots]
    if shuffle:
        random.shuffle(indices)
    totals = np.zeros(3)
    nqueries = 0
    for index in indices:
        snapshot = snapshots[index]
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(*builder(snapshots[:index], device))
        triples = augment_inverse(
            torch.as_tensor(snapshot.triples, device=device), model.num_relations
        )
        permutation = torch.randperm(len(triples), device=device)
        batches = triples[permutation].split(batch_size)
        for i, batch in enumerate(batches):
            loss, entity_loss, relation_loss = model.loss(state, batch, entity_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at timestamp {snapshot.time}")

            (loss * (len(batch) / len(triples))).backward(
                retain_graph=i < len(batches) - 1
            )
            totals += np.array(
                [loss.item(), entity_loss.item(), relation_loss.item()]
            ) * len(batch)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        nqueries += len(triples)
    if not nqueries:
        raise ValueError("Training requires at least two snapshots")
    return dict(
        zip(("loss", "entity_loss", "relation_loss"), (totals / nqueries).tolist()),
        queries=nqueries,
        snapshots=len(indices),
        last_grad_norm=float(norm),
    )


def save_checkpoint(path, model, optimizer, epoch, best_mrr, args, audit, stale=0):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "num_entities": len(model.dynamic_emb),
        "num_relations": model.num_relations,
        "epoch": epoch,
        "best_valid_raw_mrr": best_mrr,
        "args": vars(args),
        "dataset_audit": audit,
        "source_sha256": source_hashes(),
        "stale_evaluations": stale,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])
    if config.pop("decoder", "convtranse") != "convtranse":
        raise ValueError("Only two-channel ConvTransE checkpoints are supported")
    config.pop("time_origin", None)
    config.pop("time_scale", None)
    checkpoint["model_config"] = asdict(ModelConfig(**config))
    return checkpoint


def parser():
    p = argparse.ArgumentParser(
        description="Train and evaluate TKG-LDG with a two-channel decoder."
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--dataset", "-d", choices=PRESETS, default="ICEWS14")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--data-name",
        help="Actual data subdirectory; ICEWS14 prefers TiRGN's paper-matching ICEWS14s",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--evaluate-every", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--entity-weight", type=float, default=0.7)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--long-window", type=int)
    p.add_argument("--short-window", type=int)
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--long-layers", type=int, default=2)
    p.add_argument("--short-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--channels", type=int, default=50)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--graph-cache-size", type=int, default=64)
    p.add_argument("--max-train-snapshots", type=int)
    p.add_argument("--max-eval-snapshots", type=int)
    p.add_argument("--ordered-training", action="store_true")
    p.add_argument(
        "--resume", type=Path, help="Resume last.pt, preserving optimizer and RNG state"
    )
    p.add_argument("--test", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--skip-test", action="store_true")
    return p


def parse_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text())
        unknown = set(defaults) - {action.dest for action in p._actions}
        if unknown:
            p.error(f"Unknown config options: {sorted(unknown)}")
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    args.data_root = Path(args.data_root)
    args.output = Path(args.output or ROOT / "runs" / f"{args.dataset}_seed{args.seed}")
    args.long_window = (
        PRESETS[args.dataset][0] if args.long_window is None else args.long_window
    )
    args.short_window = (
        PRESETS[args.dataset][1] if args.short_window is None else args.short_window
    )
    if (
        min(args.long_window, args.short_window) < 0
        or args.long_window + args.short_window == 0
    ):
        p.error("At least one nonnegative history window must be positive")
    for name in (
        "batch_size",
        "eval_batch_size",
        "epochs",
        "evaluate_every",
        "num_threads",
        "graph_cache_size",
    ):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.lr <= 0 or args.grad_clip <= 0 or not 0 <= args.entity_weight <= 1:
        p.error("Invalid learning rate, gradient clip, or entity weight")
    for name in ("max_train_snapshots", "max_eval_snapshots"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.test and args.resume:
        p.error("--test and --resume cannot be combined")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    if device.type == "cuda":

        torch.cuda.set_device(device)
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    if args.data_name is None:
        args.data_name = (
            "ICEWS14s"
            if args.dataset == "ICEWS14" and (args.data_root / "ICEWS14s").is_dir()
            else args.dataset
        )
    ne, nr, splits = load_dataset(args.data_root, args.data_name)
    audit = audit_dataset(args.data_root, args.data_name, splits, ne, nr)
    c = ModelConfig(
        args.dim,
        args.long_layers,
        args.short_layers,
        args.dropout,
        args.channels,
        args.kernel_size,
        not args.no_normalize,
    )
    checkpoint_path = args.checkpoint or args.output / "best.pt"
    restored = None
    if args.test or args.resume:
        restored = load_checkpoint(
            checkpoint_path if args.test else args.resume, device
        )
        if (ne, nr) != (
            restored["num_entities"],
            restored["num_relations"],
        ) or args.dataset != restored["args"]["dataset"]:
            raise ValueError("Checkpoint belongs to a different dataset")
        for split in splits:
            if (
                audit["splits"][split]["sha256"]
                != restored["dataset_audit"]["splits"][split]["sha256"]
            ):
                raise ValueError(f"{split} data differs from checkpoint")
        if args.test:
            c = ModelConfig(**restored["model_config"])
            args.long_window = restored["args"]["long_window"]
            args.short_window = restored["args"]["short_window"]
        else:
            if asdict(c) != restored["model_config"]:
                raise ValueError("Resume model configuration differs from checkpoint")
            for name in (
                "data_name",
                "seed",
                "long_window",
                "short_window",
                "entity_weight",
                "lr",
                "batch_size",
                "grad_clip",
                "ordered_training",
                "max_train_snapshots",
                "max_eval_snapshots",
            ):
                if getattr(args, name) != restored["args"][name]:
                    raise ValueError(f"Resume configuration differs: {name}")
    builder = HistoryGraphBuilder(
        ne, nr, args.long_window, args.short_window, args.graph_cache_size
    )
    model = TKGLDG(ne, nr, c).to(device)
    if restored:
        model.load_state_dict(restored["model"])
    args.output.mkdir(parents=True, exist_ok=True)

    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, str(value.resolve()))
    output = Path(args.output)
    if args.test:
        result = evaluate(
            model,
            builder,
            splits["train"] + splits["valid"],
            splits["test"],
            device,
            args.eval_batch_size,
            args.max_eval_snapshots,
        )
        result.update(
            checkpoint=str(Path(checkpoint_path).resolve()),
            checkpoint_epoch=restored["epoch"],
            limited_evaluation=bool(args.max_eval_snapshots),
            dataset=args.dataset,
        )
        write_json(output / "test_metrics.json", result)
        print(json.dumps(result), flush=True)
        return result
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError(
            f"Run already exists: {output}. Choose a new output or use --resume."
        )
    metadata = {
        "args": vars(args),
        "model_config": asdict(c),
        "dataset": audit,
        "versions": {
            "torch": torch.__version__,
            "dgl": dgl.__version__,
            "python": sys.version,
        },
        "source_sha256": source_hashes(),
        "selection_metric": "valid.raw.mrr",
        "limited_run": bool(args.max_train_snapshots or args.max_eval_snapshots),
        "command": sys.argv,
        "pid": os.getpid(),
    }
    write_json(output / "run.json", metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best, stale = 1, -1.0, 0
    if args.resume:
        optimizer.load_state_dict(restored["optimizer"])
        start, best = restored["epoch"] + 1, restored["best_valid_raw_mrr"]
        stale = restored.get("stale_evaluations", 0)
        rng = restored["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])
        if not (output / "best.pt").exists():
            raise ValueError(
                "Resume in the original output directory so best.pt remains available"
            )
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset": audit,
                "model": asdict(c),
                "windows": [args.long_window, args.short_window],
            }
        ),
        flush=True,
    )
    for epoch in range(start, args.epochs + 1):
        then = time.monotonic()
        training = train_epoch(
            model,
            optimizer,
            builder,
            splits["train"],
            device,
            args.batch_size,
            args.entity_weight,
            args.grad_clip,
            args.max_train_snapshots,
            not args.ordered_training,
        )
        record = {"epoch": epoch, "train": training}
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            validation = evaluate(
                model,
                builder,
                splits["train"],
                splits["valid"],
                device,
                args.eval_batch_size,
                args.max_eval_snapshots,
            )
            record["valid"] = validation
            mrr = validation["raw"]["mrr"]
            if mrr > best:
                best, stale = mrr, 0
                save_checkpoint(
                    output / "best.pt", model, optimizer, epoch, best, args, audit
                )
            else:
                stale += 1
        record.update(seconds=time.monotonic() - then, best_valid_raw_mrr=best)
        save_checkpoint(
            output / "last.pt", model, optimizer, epoch, best, args, audit, stale
        )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.patience > 0 and stale >= args.patience:
            break
    result = {"best_valid_raw_mrr": best, "limited_run": metadata["limited_run"]}
    if not args.skip_test:
        selected = load_checkpoint(output / "best.pt", device)
        model.load_state_dict(selected["model"])
        result["test"] = evaluate(
            model,
            builder,
            splits["train"] + splits["valid"],
            splits["test"],
            device,
            args.eval_batch_size,
            args.max_eval_snapshots,
        )
        result["checkpoint_epoch"] = selected["epoch"]
    write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
