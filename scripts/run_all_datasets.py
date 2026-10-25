import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ICEWS14", "YAGO", "WIKI", "ICEWS18", "ICEWS05-15", "GDELT")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train selected datasets and seeds on one or more devices."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "batch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args(argv)
    for name in ("datasets", "seeds", "devices"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name} are not supported")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = Queue()
    for dataset in args.datasets:
        for seed in args.seeds:
            run = output / f"{dataset}_seed{seed}"
            if run.exists():
                parser.error(f"Run already exists: {run}; use a new output root")
            jobs.put((dataset, seed, run))

    def worker(device):
        outcomes = []
        while True:
            try:
                dataset, seed, run = jobs.get_nowait()
            except Empty:
                return outcomes
            run.mkdir()
            command = [
                sys.executable,
                "-u",
                str(ROOT / "main.py"),
                "--config",
                str(ROOT / "configs" / f"{dataset}.json"),
                "--data-root",
                str(args.data_root.resolve()),
                "--device",
                device,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--output",
                str(run),
            ]
            print(f"Starting {dataset}, seed={seed}, device={device}", flush=True)
            with (run / "console.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = run / "result.json"
            success = completed.returncode == 0 and result.exists()
            if success:
                record = json.loads(result.read_text())
                success = not record.get("limited_run", True) and "test" in record
            outcomes.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "device": device,
                    "status": "completed" if success else "failed",
                    "exit_code": completed.returncode,
                    "result": str(result),
                }
            )
            print(
                f"Finished {dataset}, seed={seed}: {outcomes[-1]['status']}", flush=True
            )
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        workers = [pool.submit(worker, device) for device in args.devices]
        outcomes = [
            outcome for worker_result in workers for outcome in worker_result.result()
        ]
    (output / "batch.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    results = [item["result"] for item in outcomes if item["status"] == "completed"]
    if results:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "summarize.py"),
                *results,
                "--output",
                str(output / "summary.csv"),
            ],
            check=True,
        )
    return 0 if all(item["status"] == "completed" for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
