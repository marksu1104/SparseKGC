import argparse
import csv
import os
import shlex
import subprocess
import sys
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

DEFAULT_ARGS = {
    "num_paths_to_collect": 1000,
    "max_path_len": 3,
    "prevent_loops": 1,
    "max_num_programs": 60,
    "k_adj": 40,
    "linkage": 0.25,
    "use_path_counts": 1,
}

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXEC = sys.executable


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def ensure_data(data_root, datasets, dry_run=False):
    missing = [ds for ds in datasets if not (data_root / "data" / ds / "graph.txt").exists()]
    if not missing:
        return
    cmd = [
        PYTHON_EXEC,
        "prepare_sparsekgc_data.py",
        "--output_root",
        str(data_root),
        "--datasets",
        *missing,
    ]
    print("Preparing Prob-CBR data:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=BASE_DIR, check=True)


def build_command(dataset, data_root, expt_root, args, test=False, only_preprocess=False):
    name = f"probcbr_{dataset}_{args.max_path_len}hop_{args.num_paths_to_collect}paths"
    cmd = [
        PYTHON_EXEC,
        "-u",
        "-m",
        "prob_cbr.pr_cbr",
        "--dataset_name",
        dataset,
        "--data_dir",
        str(data_root),
        "--expt_dir",
        str(expt_root),
        "--num_paths_to_collect",
        str(args.num_paths_to_collect),
        "--max_path_len",
        str(args.max_path_len),
        "--prevent_loops",
        str(args.prevent_loops),
        "--max_num_programs",
        str(args.max_num_programs),
        "--k_adj",
        str(args.k_adj),
        "--linkage",
        str(args.linkage),
        "--use_path_counts",
        str(args.use_path_counts),
        "--name_of_run",
        name,
        "--use_wandb",
        "0",
    ]
    if test:
        cmd.append("--test")
    if only_preprocess:
        cmd.append("--only_preprocess")
    return cmd


def output_root():
    root = os.environ.get("SPARSEKGC_OUTPUT_DIR")
    return Path(root) if root else None


def baseline_output_dir():
    root = output_root()
    path = root / "probcbr" if root else BASE_DIR / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_timing(row):
    root = output_root()
    timing_dir = root / "probcbr" if root else BASE_DIR / "timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    path = timing_dir / "probcbr_timings.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "baseline", "model", "dataset", "status",
            "seconds", "log_file", "command"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_with_tee(cmd, log_file, dataset, dry_run=False, params_str=""):
    command_str = " ".join(shlex.quote(x) for x in cmd)
    print_start(timestamp(), "Prob-CBR", "Prob-CBR", dataset, params_str)
    if dry_run:
        return {"status": "dry_run", "seconds": 0.0, "log_file": str(log_file), "command": command_str}
    log_file.parent.mkdir(parents=True, exist_ok=True)
    status = "ok"
    start = time.perf_counter()
    with log_file.open("w", buffering=1) as f:
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
            status = "failed"
    seconds = time.perf_counter() - start
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": "Prob-CBR",
        "model": "Prob-CBR",
        "dataset": dataset,
        "status": status,
        "seconds": f"{seconds:.3f}",
        "log_file": str(log_file),
        "command": command_str,
    }
    append_timing(row)
    if status == "ok":
        final_line = None
        with log_file.open("r", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip().strip("[]")
                if "FINAL_EVAL_METRICS" in line:
                    final_line = line
        print_result(timestamp(), "Prob-CBR", "Prob-CBR", dataset, log_file, None, final_line, seconds, status)
    if status != "ok":
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run Prob-CBR on SparseKGC datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data_root", default="prob-cbr-data")
    # If caller did not provide --expt_root, prefer SPARSEKGC_OUTPUT_DIR/probcbr when available
    parser.add_argument("--expt_root", default=None)
    parser.add_argument("--num_paths_to_collect", type=int, default=DEFAULT_ARGS["num_paths_to_collect"])
    parser.add_argument("--max_path_len", type=int, default=DEFAULT_ARGS["max_path_len"])
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=DEFAULT_ARGS["prevent_loops"])
    parser.add_argument("--max_num_programs", type=int, default=DEFAULT_ARGS["max_num_programs"])
    parser.add_argument("--k_adj", type=int, default=DEFAULT_ARGS["k_adj"])
    parser.add_argument("--linkage", type=float, default=DEFAULT_ARGS["linkage"])
    parser.add_argument("--use_path_counts", type=int, choices=[0, 1], default=DEFAULT_ARGS["use_path_counts"])
    parser.add_argument("--test", action="store_true", help="Evaluate on test instead of dev")
    parser.add_argument("--only_preprocess", action="store_true", help="Build caches without running final CBR eval")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    # determine expt_root: prefer explicit arg, else SPARSEKGC_OUTPUT_DIR/probcbr, else default
    if args.expt_root:
        expt_root = Path(args.expt_root).resolve()
    else:
        sp_out = os.environ.get("SPARSEKGC_OUTPUT_DIR")
        if sp_out:
            expt_root = Path(sp_out) / "probcbr"
        else:
            expt_root = Path("prob-cbr-expts").resolve()
    ensure_data(data_root, args.datasets, dry_run=args.dry_run)

    for dataset in args.datasets:
        cmd = build_command(dataset, data_root, expt_root, args, test=args.test, only_preprocess=args.only_preprocess)
        log_file = baseline_output_dir() / f"Prob-CBR_{dataset}.log"
        params_str = " ".join(f"{k}={getattr(args, k)}" for k in DEFAULT_ARGS)
        run_with_tee(cmd, log_file, dataset, dry_run=args.dry_run, params_str=params_str)


if __name__ == "__main__":
    main()
