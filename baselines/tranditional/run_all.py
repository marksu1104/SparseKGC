import os
import subprocess
import argparse
import sys
import csv
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from log_format import print_start, print_result

MODELS = ["TransE", "DistMult", "ComplEx", "ConvE", "TuckER", "RotatE"]
DATASETS = [
    "WD-singer",
    "FB15K-237-10",
    "WN18RR",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "FB15K-237",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

DEFAULT_ARGS = {
    "max_epochs": 500,
    "emb_dim": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "gpu": 0,
    "patience": 25,
    "eval_freq": 1,
}

# Per-model literature-common settings (shared across all datasets).
# Training-budget knobs (max_epochs, emb_dim, patience, eval_freq, gpu) stay
# unified via DEFAULT_ARGS / CLI args for fair comparison; only the
# model-specific optimization hyperparameters below are overridden.
# Sources: Dettmers et al. 2018 ConvE official config (lr=0.003, batch=128),
#          Balazevic et al. 2019 TuckER official config (lr=0.0005, batch=128).
# NOTE: ComplEx/TransE/RotatE/ConvE/TuckER use l2=0.0 (not the literature-common
# 0.01) because this implementation applies l2 as Adam weight_decay directly on
# the embeddings -- weight_decay=0.01 collapses them to an all-zero degenerate
# fixed point (loss stuck at ln(2)) within the first epoch.
# DistMult uses a small l2=1e-5: with l2=0.0 and no other regularization,
# DistMult on small datasets (e.g. WD-singer) is highly seed-sensitive --
# training loss can collapse to ~0 within ~6 epochs while valid MRR collapses
# in the same window, depending on random init (observed MRR range 0.02-0.30
# across seeds with l2=0.0). l2=1e-5 is 1000x smaller than the collapsing
# 0.01 and provides enough regularization to avoid the early overfit collapse.
MODEL_CONFIGS = {
    "TransE": {"lr": 1e-3, "l2": 0.0, "batch_size": 256},
    "RotatE": {"lr": 1e-3, "l2": 0.0, "batch_size": 256},
    "DistMult": {"lr": 1e-3, "l2": 1e-5, "batch_size": 256},
    "ComplEx": {"lr": 1e-3, "l2": 0.0, "batch_size": 256},
    "ConvE": {"lr": 0.003, "l2": 0.0, "batch_size": 128},
    "TuckER": {"lr": 0.0005, "l2": 0.0, "batch_size": 128},
}


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def get_dataset_path(dataset_name):
    return os.path.abspath(os.path.join(BASE_DIR, f"../../datasets/{dataset_name}"))


def run_with_tee(cmd, log_file):
    with open(log_file, "w", buffering=1) as f:
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            f.write(line)
            f.flush()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def read_result_lines(log_file):
    valid_line = None
    final_line = None
    with open(log_file, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if " valid]: MRR:" in line:
                valid_line = line
            if "FINAL_EVAL_METRICS" in line:
                final_line = line
    return valid_line, final_line


def output_root():
    return os.environ.get("SPARSEKGC_OUTPUT_DIR")


def baseline_output_dir():
    root = output_root()
    if root:
        path = os.path.join(root, "traditional")
        os.makedirs(path, exist_ok=True)
        return path
    os.makedirs("logs", exist_ok=True)
    return "logs"


def append_timing(row):
    root = output_root()
    if root:
        timing_dir = os.path.join(root, "traditional")
    else:
        timing_dir = "timings"
    os.makedirs(timing_dir, exist_ok=True)
    path = os.path.join(timing_dir, "traditional_timings.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "baseline", "model", "dataset", "status",
            "seconds", "log_file", "command"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(model, dataset, dry_run=False, **kwargs):
    data_path = get_dataset_path(dataset)
    if not os.path.exists(data_path):
        print(f"Skipping {dataset}: Path {data_path} not found.", flush=True)
        return {"status": "skipped", "seconds": 0.0, "log_file": ""}

    cmd = [PYTHON_EXEC, "-u", "main.py"]

    args = DEFAULT_ARGS.copy()
    args.update(kwargs)
    args.update(MODEL_CONFIGS.get(model, {}))

    cmd.extend(["--model", model])
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--data_path", data_path])

    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])

    if model in {"TransE", "RotatE"}:
        cmd.extend(["--margin", "9.0"])
        args["margin"] = 9.0

    command_str = " ".join(cmd)
    log_file = os.path.join(baseline_output_dir(), f"{model}_{dataset}.log")
    params_str = " ".join(f"{k}={v}" for k, v in args.items() if k != "gpu")
    print_start(timestamp(), "traditional", model, dataset, params_str)
    if dry_run:
        return {"status": "dry_run", "seconds": 0.0, "log_file": log_file, "command": command_str}

    status = "ok"
    start = time.perf_counter()
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        run_with_tee(cmd, log_file)
    except subprocess.CalledProcessError as e:
        status = "failed"
        print(f"Error running {model} on {dataset}: {e}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
        sys.exit(1)
    seconds = time.perf_counter() - start
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": "traditional",
        "model": model,
        "dataset": dataset,
        "status": status,
        "seconds": f"{seconds:.3f}",
        "log_file": log_file,
        "command": command_str,
    }
    append_timing(row)
    if status == "ok":
        valid_line, final_line = read_result_lines(log_file)
        print_result(timestamp(), "traditional", model, dataset, log_file, valid_line, final_line, seconds, status)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Experiments (aligned with run_all_baselines.sh)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="Models to include")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to include")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name (kept for compatibility)")
    parser.add_argument("--gpu", type=int, default=DEFAULT_ARGS['gpu'], help="GPU ID")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_ARGS['max_epochs'], help="Epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_ARGS['batch_size'], help="Batch size")
    parser.add_argument("--patience", type=int, default=DEFAULT_ARGS['patience'], help="Early stopping patience")
    parser.add_argument("--emb_dim", type=int, default=DEFAULT_ARGS['emb_dim'], help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=DEFAULT_ARGS['lr'], help="Learning rate")
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_ARGS['eval_freq'], help="Evaluation frequency")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")

    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else args.datasets

    for dataset in datasets:
        for model in args.models:
            run_experiment(
                model,
                dataset,
                dry_run=args.dry_run,
                gpu=args.gpu,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                emb_dim=args.emb_dim,
                lr=args.lr,
                eval_freq=args.eval_freq,
            )


if __name__ == "__main__":
    main()
