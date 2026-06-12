import argparse
import csv
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from log_format import print_start, print_result

DATASETS = [
    "WD-singer",
    "FB15K-237-10",
    "WN18RR",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "FB15K-237",
]

SCORE_FUNCS = ["conve"]

GENERIC_CONFIGS = {
    "transe": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func transe -chan_drop 0.1 -rel_norm -hid_drop 0.2 -sim_decay 1e-6",
    "distmult": "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func distmult -chamix_dim 200 -relmix_dim 200 -rel_norm -hid_drop 0.3",
    "conve": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -k_w 10 -k_h 10",
}

# Paper/legacy settings where this repo already had tuned scripts.
TUNED_CONFIGS = {
    ("NELL23K", "conve"): "-rel_reason -pre_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func conve -chan_drop 0.1 -rel_mask 0.1 -rel_norm -hid_drop 0.3",
    ("NELL23K", "distmult"): "-rel_reason -batch 256 -init_dim 150 -gcn_dim 150 -embed_dim 150 -gcn_layer 2 -gcn_drop 0.3 -score_func distmult -chan_drop 0.2 -hid_drop 0.2",
    ("NELL23K", "transe"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.2 -score_func transe -chan_drop 0.1 -rel_norm -hid_drop 0.2 -sim_decay 1e-5",
    ("WD-singer", "conve"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.1 -k_w 10 -k_h 10",
    ("WD-singer", "distmult"): "-rel_reason -batch 256 -init_dim 150 -gcn_dim 150 -embed_dim 150 -gcn_layer 2 -gcn_drop 0.0 -score_func distmult -chan_drop 0.1 -rel_mask 0.0 -rel_norm -hid_drop 0.2",
    ("WD-singer", "transe"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func transe -chan_drop 0.3 -rel_mask 0.1 -rel_norm -hid_drop 0.1",
    ("WN18RR", "conve"): "-rel_reason -reason_type mixdrop2 -bias -batch 256 -init_dim 200 -gcn_dim 200 -embed_dim 200 -gcn_layer 1 -gcn_drop 0.0 -score_func conve -chamix_dim 300 -relmix_dim 300 -rel_norm -hid_drop 0.3 -hid_drop2 0.5 -feat_drop 0.1 -k_w 10 -k_h 20 -num_filt 250 -ker_sz 7",
    ("FB15K-237-10", "conve"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5 -k_w 10 -k_h 10",
    ("FB15K-237-10", "distmult"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.3 -score_func distmult -hid_drop 0.1 -sim_decay 1e-5",
    ("FB15K-237-10", "transe"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func transe -chan_drop 0.1 -rel_norm -hid_drop 0.2 -sim_decay 1e-6",
    ("FB15K-237-20", "distmult"): "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func distmult -chamix_dim 400 -relmix_dim 400 -rel_norm -hid_drop 0.3",
    ("FB15K-237-50", "distmult"): "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.0 -score_func distmult -chamix_dim 200 -relmix_dim 200 -rel_norm -hid_drop 0.3",
    # No published tuned config for conve on FB15K-237-20/50; reuse the
    # FB15K-237-10 conve config (closest sparsity regime with a tuned entry)
    # rather than falling back to GENERIC_CONFIGS["conve"].
    ("FB15K-237-20", "conve"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5 -k_w 10 -k_h 10",
    ("FB15K-237-50", "conve"): "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5 -k_w 10 -k_h 10",
    ("FB15K-237", "distmult"): "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.0 -score_func distmult -chamix_dim 200 -relmix_dim 200 -rel_norm -hid_drop 0.3",
}


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def normalize_dataset_name(dataset):
    return "FB15K-237" if dataset == "FB15k-237" else dataset


def get_config(dataset, score_func):
    dataset = normalize_dataset_name(dataset)
    return TUNED_CONFIGS.get((dataset, score_func), GENERIC_CONFIGS[score_func])


def output_root():
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    return Path(root) if root else None


def baseline_output_dir():
    root = output_root()
    path = root / "hogrn" if root else Path("logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_timing(row):
    root = output_root()
    timing_dir = root / "hogrn" if root else Path("timings")
    timing_dir.mkdir(parents=True, exist_ok=True)
    path = timing_dir / "hogrn_timings.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "baseline", "model", "dataset", "status",
            "seconds", "log_file", "command"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_data_root():
    root = os.environ.get("SPARSEKGC_DATA_DIR", "../../datasets")
    return Path(root).resolve()


def run_one(dataset, score_func, gpu, dry_run=False):
    if not Path("run.py").exists():
        raise SystemExit("run.py not found; run this script from baselines/HoGRN")
    data_root = resolve_data_root()
    data_dir = data_root / dataset
    if not data_dir.exists() and dataset == "FB15K-237" and (data_root / "FB15k-237").exists():
        dataset = "FB15k-237"
        data_dir = data_root / dataset
    if not data_dir.exists():
        print(f"Skipping HoGRN {score_func} {dataset}: missing {data_dir}")
        return {"status": "skipped", "seconds": 0.0}

    args = shlex.split(get_config(dataset, score_func))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir = baseline_output_dir()
    Path("checkpoints").mkdir(exist_ok=True)
    log_file = log_dir / f"HoGRN_{score_func}_{normalize_dataset_name(dataset)}.log"

    params_str = get_config(dataset, score_func)
    print_start(timestamp(), "HoGRN", score_func, normalize_dataset_name(dataset), params_str)

    status = "ok"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hogrn_log_") as tmp_logdir:
        cmd = [
            sys.executable,
            "-u",
            "run.py",
            "-data",
            dataset,
            "-data_root",
            str(data_root),
            "-logdir",
            tmp_logdir,
            *args,
        ]
        command_str = " ".join(shlex.quote(x) for x in cmd)
        if dry_run:
            return {"status": "dry_run", "seconds": 0.0, "log_file": str(log_file), "command": command_str}

        with log_file.open("w", buffering=1) as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in process.stdout:
                f.write(line)
                f.flush()
            process.wait()
            if process.returncode != 0:
                status = "failed"
    seconds = time.perf_counter() - start
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": "HoGRN",
        "model": score_func,
        "dataset": normalize_dataset_name(dataset),
        "status": status,
        "seconds": f"{seconds:.3f}",
        "log_file": str(log_file),
        "command": command_str,
    }
    append_timing(row)
    if status == "ok":
        valid_line = None
        final_line = None
        with log_file.open("r", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if " valid]: MRR:" in line:
                    valid_line = line
                if "FINAL_EVAL_METRICS" in line:
                    final_line = line
        print_result(timestamp(), "HoGRN", score_func, normalize_dataset_name(dataset), log_file, valid_line, final_line, seconds, status)
    if status != "ok":
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run HoGRN on multiple datasets and score functions")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--score_funcs", nargs="+", default=SCORE_FUNCS, choices=SCORE_FUNCS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        for score_func in args.score_funcs:
            run_one(dataset, score_func, args.gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
