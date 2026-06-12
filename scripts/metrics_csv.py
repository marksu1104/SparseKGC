"""Shared upsert helper for the per-baseline metrics CSVs.

All four baselines (traditional, HoGRN, Prob-CBR, DacKGR) write rows keyed by
(Dataset, Model) into outputs/{baseline}_metrics.csv using the same header
(see METRICS_CSV_HEADER below). upsert_metrics_csv() replaces any existing
row with the same key instead of appending a duplicate, so reruns always
leave the latest result for each (Dataset, Model) combo.
"""

import csv
import os

METRICS_CSV_HEADER = [
    "Dataset", "Model",
    "MRR_Tail", "MRR_Head", "MRR_Avg",
    "Hits@1_Tail", "Hits@1_Head", "Hits@1_Avg",
    "Hits@3_Tail", "Hits@3_Head", "Hits@3_Avg",
    "Hits@10_Tail", "Hits@10_Head", "Hits@10_Avg",
    "seconds",
]


def upsert_metrics_csv(path, row, header=METRICS_CSV_HEADER, key_cols=("Dataset", "Model")):
    """Write `row` (a list matching `header`) into the CSV at `path`,
    replacing any existing row with the same (Dataset, Model) key."""
    key_idx = [header.index(c) for c in key_cols]
    key = tuple(str(row[i]) for i in key_idx)

    rows = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            for existing_row in reader:
                if not existing_row:
                    continue
                existing_key = tuple(str(existing_row[i]) for i in key_idx)
                if existing_key != key:
                    rows.append(existing_row)

    rows.append([str(v) for v in row])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
