"""Shared, human-readable log formatting for SparseKGC baseline runners.

Used by baselines/tranditional/run_all.py, baselines/HoGRN/run_hogrn_all.py,
and baselines/Prob-CBR/run_all.py so that all three baselines print results
in the same compact layout instead of each using its own ad-hoc separators.
"""

import re

_VALID_RE = re.compile(
    r"\[Epoch (\d+) valid\]:\s*MRR:\s*Tail\s*:\s*([\d.]+),\s*Head\s*:\s*([\d.]+),\s*Avg\s*:\s*([\d.]+)"
)


def parse_metrics(line):
    """Extract the key=value pairs from a FINAL_EVAL_METRICS line."""
    if not line:
        return None
    idx = line.find("FINAL_EVAL_METRICS")
    if idx == -1:
        return None
    metrics = {}
    for token in line[idx:].split():
        if "=" in token:
            key, value = token.split("=", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    return metrics or None


def parse_valid(line):
    """Extract (epoch, tail, head, avg) MRR from a '[Epoch N valid]: MRR: ...' line."""
    if not line:
        return None
    m = _VALID_RE.search(line)
    if not m:
        return None
    return {
        "epoch": int(m.group(1)),
        "tail": float(m.group(2)),
        "head": float(m.group(3)),
        "avg": float(m.group(4)),
    }


def print_start(now_str, baseline, model, dataset, params=""):
    print(f"[{now_str}] {baseline}/{model}/{dataset}  START", flush=True)
    if params:
        print(f"    params : {params}", flush=True)


def print_result(now_str, baseline, model, dataset, log_file, valid_line, final_line, seconds, status="ok"):
    state = "DONE" if status == "ok" else "FAILED"
    print(f"[{now_str}] {baseline}/{model}/{dataset}  {state} ({seconds:.1f}s)", flush=True)

    valid = parse_valid(valid_line)
    if valid:
        print(
            f"    valid (epoch {valid['epoch']:>4}) : MRR avg={valid['avg']:.4f}"
            f"  (tail={valid['tail']:.4f} / head={valid['head']:.4f})",
            flush=True,
        )
    else:
        print("    valid           : not_run", flush=True)

    metrics = parse_metrics(final_line)
    if metrics:
        print(
            f"    test            : MRR avg={metrics.get('mrr_avg', float('nan')):.4f}"
            f"  (tail={metrics.get('mrr_tail', float('nan')):.4f} / head={metrics.get('mrr_head', float('nan')):.4f})",
            flush=True,
        )
        print(
            f"                      H@1 avg={metrics.get('h1_avg', float('nan')):.4f}"
            f"  H@3 avg={metrics.get('h3_avg', float('nan')):.4f}"
            f"  H@10 avg={metrics.get('h10_avg', float('nan')):.4f}",
            flush=True,
        )
    else:
        print("    test            : not_found", flush=True)

    print(f"    log: {log_file}", flush=True)
    print("-" * 80, flush=True)
