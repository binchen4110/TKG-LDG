import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = ("mrr", "hits@1", "hits@3", "hits@10")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize completed raw test metrics in percent."
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows, groups = [], {}
    for path in args.results:
        result = json.loads(path.read_text())
        run = json.loads((path.parent / "run.json").read_text())
        if result.get("limited_run", True) or "test" not in result:
            parser.error(f"Incomplete or truncated result: {path}")
        config = run["args"]
        dataset, seed = config["dataset"], config["seed"]
        signature = {
            key: config[key]
            for key in (
                "long_window",
                "short_window",
                "entity_weight",
                "lr",
                "batch_size",
                "grad_clip",
                "ordered_training",
                "epochs",
                "patience",
                "evaluate_every",
            )
        }
        signature.update(
            model_config=run["model_config"],
            dataset=run["dataset"],
            source_sha256=run["source_sha256"],
        )
        if dataset not in groups:
            groups[dataset] = {"signature": signature, "seeds": set(), "values": []}
        group = groups[dataset]
        if signature != group["signature"]:
            parser.error(
                f"Cannot aggregate different data, source, or settings for {dataset}"
            )
        if seed in group["seeds"]:
            parser.error(f"Duplicate seed {seed} for {dataset}")
        group["seeds"].add(seed)
        values = {key: result["test"]["raw"][key] * 100 for key in METRICS}
        group["values"].append(values)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "best_epoch": result["checkpoint_epoch"],
                **values,
            }
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("dataset", "seed", "best_epoch", *METRICS)
            )
            writer.writeheader()
            writer.writerows(rows)
    summary = {"units": "percent", "datasets": {}}
    for dataset, group in groups.items():
        metrics = {}
        for key in METRICS:
            values = [item[key] for item in group["values"]]
            metrics[key] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        summary["datasets"][dataset] = {"seeds": sorted(group["seeds"]), "raw": metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
