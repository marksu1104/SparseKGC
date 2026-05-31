<<<<<<< HEAD
import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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


def run_one(dataset, score_func, gpu, dry_run=False):
    if not Path("run.py").exists():
        raise SystemExit("run.py not found; run this script from baselines/HoGRN")
    data_dir = Path("data") / dataset
    if not data_dir.exists() and dataset == "FB15K-237" and Path("data/FB15k-237").exists():
        dataset = "FB15k-237"
        data_dir = Path("data") / dataset
    if not data_dir.exists():
        print(f"Skipping HoGRN {score_func} {dataset}: missing {data_dir}")
        return {"status": "skipped", "seconds": 0.0}

    args = shlex.split(get_config(dataset, score_func))
    cmd = [sys.executable, "-u", "run.py", "-data", dataset, *args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir = baseline_output_dir()
    Path("checkpoints").mkdir(exist_ok=True)
    log_file = log_dir / f"HoGRN_{score_func}_{normalize_dataset_name(dataset)}.log"

    command_str = " ".join(shlex.quote(x) for x in cmd)
    print("=" * 72, flush=True)
    print(f"Time   | {timestamp()}", flush=True)
    print(f"Start  | baseline=HoGRN | model={score_func} | dataset={normalize_dataset_name(dataset)}", flush=True)
    print(f"Params | {command_str}", flush=True)
    print(f"Log    | {log_file}", flush=True)
    print("=" * 72, flush=True)
    if dry_run:
        return {"status": "dry_run", "seconds": 0.0, "log_file": str(log_file), "command": command_str}

    status = "ok"
    start = time.perf_counter()
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
                if "EVAL_STD model=HoGRN split=valid" in line:
                    valid_line = line
                if line.startswith("FINAL_EVAL_METRICS"):
                    final_line = line
        print("-" * 72, flush=True)
        print(f"Time   | {timestamp()}", flush=True)
        print(f"Result | baseline=HoGRN | model={score_func} | dataset={normalize_dataset_name(dataset)}", flush=True)
        print(f"  valid  : {valid_line.replace('EVAL_STD', 'FINAL_VALID_METRICS', 1) if valid_line else 'not_found'}")
        print(f"  holdout: {final_line or 'not_found'}")
        print(f"  seconds: {seconds:.3f}", flush=True)
        print(f"  log    : {log_file}", flush=True)
        print("-" * 72, flush=True)
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
=======
import os
import subprocess
from pathlib import Path

# 準備要連續執行的目標資料集
datasets = ["NELL23K", "WD-singer", "WN18RR", "FB15K-237-10", "FB15K-237-20", "FB15K-237-50"]

# 定義每個資料集在 HoGRN (ConvE 或是 DistMult) 的最佳官方參數設定
# 根據 baselines/HoGRN/sh 內部的腳本提取
configs = {
    "NELL23K": "-rel_reason -pre_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 1 -gcn_drop 0.1 -score_func conve -chan_drop 0.1 -rel_mask 0.1 -rel_norm -hid_drop 0.3",
    "WD-singer": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5",
    "WN18RR": "-rel_reason -reason_type 'mixdrop2' -bias -batch 256 -init_dim 200 -gcn_dim 200 -embed_dim 200 -gcn_layer 1 -gcn_drop 0.0 -score_func conve -chamix_dim 300 -relmix_dim 300 -rel_norm -hid_drop 0.3 -hid_drop2 0.5 -feat_drop 0.1 -k_w 10 -k_h 20 -num_filt 250 -ker_sz 7",
    "FB15K-237-10": "-rel_reason -batch 128 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.1 -score_func conve -chan_drop 0.2 -rel_mask 0.2 -rel_norm -hid_drop 0.3 -sim_decay 1e-5",
    "FB15K-237-20": "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func distmult -chamix_dim 400 -relmix_dim 400 -rel_norm -hid_drop 0.3",
    "FB15K-237-50": "-rel_reason -batch 256 -init_dim 100 -gcn_dim 100 -embed_dim 100 -gcn_layer 2 -gcn_drop 0.2 -score_func distmult -chamix_dim 400 -relmix_dim 400 -rel_norm -hid_drop 0.3"
}

def main():
    # 因為腳本已經搬到 HoGRN 底下，我們可以直接確認當前目錄並執行
    target_dir = Path(".")
    
    # 也可以透過檢查 run.py 是不是在當前目錄來防呆
    if not (target_dir / "run.py").exists():
        print(f"錯誤: 找不到 run.py，請確認是否在 HoGRN 目錄下執行此腳本")
        return

    # 不再需要切換工作目錄了
    # os.chdir(target_dir)

    # 開始遍歷 6 個資料集並執行
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"🚀 開始執行 HoGRN 訓練與驗證: {ds}")
        print(f"{'='*60}")
        
        args = configs.get(ds, configs["FB15K-237-10"]) # 如果沒有就拿 FB15K 的當預設兜底
        
        # 組裝執行指令： CUDA_VISIBLE_DEVICES=0 python run.py -data {ds} {args}
        cmd = f"CUDA_VISIBLE_DEVICES=0 python run.py -data {ds} {args}"
        
        print(f"執行的 Command: {cmd}\n")
        
        # 開啟 log 儲存，以便我們後補擷取 (導向 stderr 到 stdout，一起存)
        log_file = f"HoGRN_{ds}.log"
        full_cmd = f"{cmd} 2>&1 | tee {log_file}"
        
        try:
            # 加上 stdout 導向，讓使用者能清楚看見運作過程
            subprocess.run(full_cmd, shell=True, check=True)
            print(f"\n✅ {ds} 執行完成！")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {ds} 執行失敗，錯誤碼: {e.returncode}")

if __name__ == '__main__':
    main()
    
    # 執行所有實驗後，自動進行結果整理
    print("\n📦 開始整理 HoGRN 測試結果...")
    try:
        subprocess.run("python aggregate_hogrn_results.py", shell=True, check=True)
    except Exception as e:
        print(f"整理結果失敗: {e}")

    main()
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
