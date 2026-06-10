import argparse
import shutil
from pathlib import Path
from prob_cbr.data.data_utils import get_inv_relation

DATASETS = [
    "WD-singer",
    "FB15K-237-10",
    "WN18RR",
    "FB15K-237-20",
    "FB15K-237-50",
    "NELL23K",
    "FB15K-237",
]


def convert_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, tail, relation = parts[:3]
            fout.write(f"{head}\t{relation}\t{tail}\n")


def convert_file_with_inverse(src, dst, dataset):
    """Write original + inverse edges so CBR can traverse the graph bidirectionally."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, tail, relation = parts[:3]
            fout.write(f"{head}\t{relation}\t{tail}\n")
            r_inv = get_inv_relation(relation, dataset)
            fout.write(f"{tail}\t{r_inv}\t{head}\n")


def prepare_dataset(source_root, output_root, dataset):
    src_dir = source_root / dataset
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    data_dir = output_root / "data" / dataset
    subgraph_dir = output_root / "subgraphs" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    subgraph_dir.mkdir(parents=True, exist_ok=True)

    convert_file(src_dir / "train.txt", data_dir / "train.txt")
    convert_file_with_inverse(src_dir / "train.txt", data_dir / "graph.txt", dataset)
    convert_file(src_dir / "valid.txt", data_dir / "dev.txt")
    convert_file(src_dir / "test.txt", data_dir / "test.txt")


def main():
    parser = argparse.ArgumentParser(description="Prepare SparseKGC datasets for Prob-CBR")
    parser.add_argument("--source_root", default="../../datasets")
    parser.add_argument("--output_root", default="prob-cbr-data")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    for dataset in args.datasets:
        prepare_dataset(source_root, output_root, dataset)
        print(f"Prepared {dataset} -> {output_root / 'data' / dataset}")


if __name__ == "__main__":
    main()
