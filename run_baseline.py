import argparse
import os
import shlex
import subprocess
import threading
import time
import sys
from pathlib import Path
from datetime import datetime


DATASETS = [
    "WD-singer",
    "FB15K-237-10",
    "WN18RR",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "FB15K-237",
]

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

BASELINE_METRICS = {
    "traditional": OUTPUT_DIR / "traditional_metrics.csv",
    "hogrn": OUTPUT_DIR / "hogrn_metrics.csv",
    "dackgr": OUTPUT_DIR / "dackgr_metrics.csv",
    "probcbr": OUTPUT_DIR / "probcbr_metrics.csv",
}

DACKGR_PROCESS_CONFIGS = {
    "WD-singer": "configs/wd-singer.sh",
    "FB15K-237-10": "configs/fb15k-237-10.sh",
    "WN18RR": "configs/wn18rr.sh",
    "FB15K-237-20": "configs/fb15k-237-20.sh",
    "FB15K-237-50": "configs/fb15k-237-50.sh",
    "NELL23K": "configs/nell23k.sh",
    "FB15K-237": "configs/fb15k-237.sh",
}

DACKGR_CONVE_CONFIGS = {
    "WD-singer": "configs/wd-singer-conve.sh",
    "FB15K-237-10": "configs/fb15k-237-10-conve.sh",
    "WN18RR": "configs/wn18rr-conve.sh",
    "FB15K-237-20": "configs/fb15k-237-20-conve.sh",
    "FB15K-237-50": "configs/fb15k-237-50-conve.sh",
    "NELL23K": "configs/nell23k-conve.sh",
    "FB15K-237": "configs/fb15k-237-conve.sh",
}

DACKGR_RS_CONFIGS = {
    "WD-singer": "configs/wd-singer-rs.sh",
    "FB15K-237-10": "configs/fb15k-237-10-rs.sh",
    "WN18RR": "configs/wn18rr-rs.sh",
    "FB15K-237-20": "configs/fb15k-237-20-rs.sh",
    "FB15K-237-50": "configs/fb15k-237-50-rs.sh",
    "NELL23K": "configs/nell23k-rs.sh",
    "FB15K-237": "configs/fb15k-237-rs.sh",
}


def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"


def log_status(message):
    print(f"[{timestamp()}] {message}", flush=True)


def log_field(label, value):
    print(f"    {label:<8} {value}", flush=True)


def python_bin() -> str:
    return os.environ.get("PYTHON_BIN", sys.executable)


def run_command(cmd, cwd, dry_run=False, log_file=None, summary_context=None, env_overrides=None, show_summary=True, heartbeat_label=None):
    if dry_run:
        return
    env = os.environ.copy()
    env["SPARSEKGC_OUTPUT_DIR"] = str(OUTPUT_DIR)
    if env_overrides:
        env.update(env_overrides)
    if log_file is None:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", buffering=1) as f:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        def copy_output():
            for line in process.stdout:
                f.write(line)
                f.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        start_time = time.monotonic()
        next_heartbeat = start_time + 600
        while process.poll() is None:
            time.sleep(30)
            now = time.monotonic()
            if heartbeat_label and now >= next_heartbeat:
                elapsed = int(now - start_time)
                log_status(f"Still running | {heartbeat_label} | elapsed={elapsed}s | log={log_file}")
                next_heartbeat = now + 600
        reader.join()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    if log_file is not None and show_summary:
        final_line = None
        valid_line = None
        runtime_seconds = None
        with log_file.open("r", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if "Dev MRR" in line or "Dev set performance" in line:
                    valid_line = line
                if line.startswith("FINAL_EVAL_METRICS"):
                    final_line = line
                if line.startswith("RUNTIME_STD") and "seconds=" in line:
                    runtime_seconds = line.rsplit("seconds=", 1)[-1]
        context = summary_context or {}
        prefix = " ".join(f"{key}={value}" for key, value in context.items())
        print("-" * 72, flush=True)
        print(f"Time   | {timestamp()}", flush=True)
        print(f"Result | {prefix}", flush=True)
        print(f"  valid  : {valid_line or 'not_found'}", flush=True)
        print(f"  holdout: {final_line or 'not_found'}", flush=True)
        print(f"  seconds: {runtime_seconds or 'not_found'}", flush=True)
        print(f"  log    : {log_file}", flush=True)
        print("-" * 72, flush=True)


def reset_metrics(baseline):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_METRICS[baseline]
    if path.exists():
        path.unlink()


def run_traditional(args):
    cmd = [
        python_bin(),
        "run_all.py",
        "--datasets",
        *args.datasets,
        "--gpu",
        str(args.gpu),
    ]
    if args.models:
        cmd.extend(["--models", *args.models])
    if args.max_epochs is not None:
        cmd.extend(["--max_epochs", str(args.max_epochs)])
    if args.dry_run:
        cmd.append("--dry_run")
    run_command(cmd, BASE_DIR / "baselines" / "tranditional")


def run_hogrn(args):
    cmd = [
        python_bin(),
        "run_hogrn_all.py",
        "--datasets",
        *args.datasets,
        "--score_funcs",
        "conve",
        "--gpu",
        str(args.gpu),
    ]
    if args.dry_run:
        cmd.append("--dry_run")
    run_command(cmd, BASE_DIR / "baselines" / "HoGRN")


def run_dackgr(args):
    cwd = BASE_DIR / "baselines" / "DacKGR"
    stages = [
        ("process_data", "point", "./experiment.sh", DACKGR_PROCESS_CONFIGS, "--process_data", {"DACKGR_WRITE_METRICS": "0"}),
        ("pretrain_conve", "conve", "./experiment-emb.sh", DACKGR_CONVE_CONFIGS, "--train", {"DACKGR_WRITE_METRICS": "0"}),
        ("train_infer", "point.rs.conve", "./experiment-rs.sh", DACKGR_RS_CONFIGS, "--train", {"DACKGR_WRITE_METRICS": "1"}),
    ]
    for dataset in args.datasets:
        for stage, model, script, config_map, action, env_overrides in stages:
            config = config_map.get(dataset)
            if config is None:
                raise SystemExit(f"No DacKGR {stage} config registered for dataset: {dataset}")
            cmd = [script, config, action, str(args.gpu)]
            log_file = OUTPUT_DIR / "dackgr" / f"DacKGR_{stage}_{dataset}.log"
            print("=" * 72, flush=True)
            print(f"Time   | {timestamp()}", flush=True)
            print(f"Start  | baseline=DacKGR | stage={stage} | model={model} | dataset={dataset}", flush=True)
            print("Params | " + shlex.join(cmd), flush=True)
            print(f"Log    | {log_file}", flush=True)
            print("=" * 72, flush=True)
            run_command(
                cmd,
                cwd,
                dry_run=args.dry_run,
                log_file=log_file,
                summary_context={"baseline": "DacKGR", "stage": stage, "model": model, "dataset": dataset},
                env_overrides=env_overrides,
                show_summary=(stage == "train_infer"),
                heartbeat_label=f"baseline=DacKGR stage={stage} model={model} dataset={dataset}",
            )
            if stage != "train_infer":
                stage_status = "dry_run" if args.dry_run else "completed"
                print(f"Time   | {timestamp()}", flush=True)
                print(f"Done   | baseline=DacKGR | stage={stage} | status={stage_status} | dataset={dataset}", flush=True)


def run_probcbr(args):
    cmd = [
        python_bin(),
        "run_all.py",
        "--datasets",
        *args.datasets,
        "--test",
    ]
    if args.dry_run:
        cmd.append("--dry_run")
    run_command(cmd, BASE_DIR / "baselines" / "Prob-CBR")


def main():
    parser = argparse.ArgumentParser(description="Run one SparseKGC baseline with a unified entry point.")
    parser.add_argument("baseline", choices=["traditional", "hogrn", "dackgr", "probcbr"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reset_metrics", action="store_true")
    parser.add_argument("--models", nargs="+", help="Traditional-only model list.")
    parser.add_argument("--max_epochs", type=int, help="Traditional-only epoch override.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SPARSEKGC_OUTPUT_DIR"] = str(OUTPUT_DIR)

    if args.reset_metrics:
        reset_metrics(args.baseline)

    if args.baseline == "traditional":
        run_traditional(args)
    elif args.baseline == "hogrn":
        run_hogrn(args)
    elif args.baseline == "dackgr":
        run_dackgr(args)
    elif args.baseline == "probcbr":
        run_probcbr(args)



if __name__ == "__main__":
    main()
