#!/usr/bin/env python3
"""
Export BSR-compatible routing/hybrid-expert prediction artifacts from trained
SparseKGC checkpoints (HoGRN, TransE, ConvE, TuckER).

SparseKGC owns training, checkpoint loading and inference; this script only
runs inference on already-trained checkpoints and writes CSV artifacts that a
separate Case-Path BSR repo can consume as routing / hybrid experts. No
training code is duplicated into the BSR repo -- it only receives CSVs.

Usage:
    python scripts/export_bsr_routing_predictions.py \
        --data-root datasets \
        --checkpoint-root checkpoints \
        --output-root external_predictions \
        --models HoGRN TransE ConvE TuckER \
        --datasets FB15K-237-10 FB15K-237-20 FB15K-237-50 NELL23K WD-singer WN18RR \
        --splits valid test \
        --top-k 200 \
        --seed 42 \
        --gpu 0
"""
import argparse
import csv
import importlib
import importlib.util
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRADITIONAL_DIR = REPO_ROOT / "baselines" / "tranditional"
HOGRN_DIR = REPO_ROOT / "baselines" / "HoGRN"

TRAD_MODELS = {"TransE", "ConvE", "TuckER"}
HOGRN_MODELS = {"HoGRN"}
HOGRN_SCORE_FUNC_CANDIDATES = ["conve", "distmult", "transe"]

EVAL_BATCH_SIZE = 256

PROTOCOL_LABEL = "bidirectional filtered ranking with reciprocal relations"
ENTITY_MAPPING_LABEL = "sorted(all entities from train+valid+test)"
RELATION_MAPPING_LABEL = "sorted(base relations) plus reverse offset"

QUERY_SUMMARY_HEADER = [
    "model", "dataset", "split", "query_index", "direction",
    "original_h", "original_r", "original_t",
    "query_h", "query_r", "query_t", "gold_entity",
    "gold_score", "filtered_rank", "top1_entity", "top1_score",
]

TOPK_HEADER = [
    "model", "dataset", "split", "query_index", "direction",
    "original_h", "original_r", "original_t",
    "query_h", "query_r", "query_t", "gold_entity",
    "candidate_rank", "candidate_entity", "candidate_score",
]

MANIFEST_HEADER = [
    "model", "dataset", "split", "mrr", "hits@1", "hits@3", "hits@10",
    "num_queries", "summary_file", "candidates_file",
    "source_repo", "source_commit", "source_config", "seed",
    "protocol", "entity_mapping", "relation_mapping", "top_k_export", "export_time_utc",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--checkpoint-root", type=str, default=None,
                        help="Optional root containing per-baseline checkpoint subdirectories "
                             "('tranditional', 'HoGRN'). Falls back to each baseline's own "
                             "checkpoints/ directory when not given or when not found here.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["HoGRN", "TransE", "ConvE", "TuckER"],
                        choices=["HoGRN", "TransE", "ConvE", "TuckER"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test", "train"])
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# BSR-compatible ID mapping & dataset loading
# ---------------------------------------------------------------------------

def read_raw_triples(dataset_dir):
    """Return {split: [(h_str, t_str, r_str), ...]} in original file order."""
    triples = {}
    for split in ("train", "valid", "test"):
        path = dataset_dir / f"{split}.txt"
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                rows.append((parts[0], parts[1], parts[2]))
        triples[split] = rows
    return triples


def build_bsr_mapping(raw_triples):
    """entity_id = sorted(all entities).index ; base_relation_id = sorted(base relations).index"""
    entities = set()
    base_relations = set()
    for rows in raw_triples.values():
        for h, t, r in rows:
            entities.add(h)
            entities.add(t)
            base_relations.add(r)
    entities = sorted(entities)
    base_relations = sorted(base_relations)
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(base_relations)}
    return entities, ent2id, base_relations, rel2id


def translate_triples(raw_triples, ent2id, rel2id):
    out = {}
    for split, rows in raw_triples.items():
        out[split] = [(ent2id[h], rel2id[r], ent2id[t]) for h, t, r in rows]
    return out


def build_true_object_filter(translated_triples, num_base_rel):
    """
    filt[(s, r)] -> set of true objects for query (s, r) in BSR id space, where
    r may be a base relation (tail-direction query) or a reverse relation
    r_base + num_base_rel (head-direction query, answered via the reverse edge).
    Built from train+valid+test, exactly the filter needed for both directions.
    """
    filt = defaultdict(set)
    for rows in translated_triples.values():
        for h, r, t in rows:
            filt[(h, r)].add(t)
            filt[(t, r + num_base_rel)].add(h)
    return filt


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_checkpoint(model_name, dataset_name, checkpoint_root):
    """Returns (checkpoint_path, score_func) or (None, None) if not found."""
    if model_name in TRAD_MODELS:
        filename = f"best_model_{model_name}_{dataset_name}.pth"
        for d in _candidate_dirs("tranditional", TRADITIONAL_DIR / "checkpoints", checkpoint_root):
            p = d / filename
            if p.exists():
                return p, None
        return None, None

    if model_name in HOGRN_MODELS:
        for sf in HOGRN_SCORE_FUNC_CANDIDATES:
            filename = f"{dataset_name}_{sf}_best"
            for d in _candidate_dirs("HoGRN", HOGRN_DIR / "checkpoints", checkpoint_root):
                p = d / filename
                if p.exists():
                    return p, sf
        return None, None

    raise ValueError(f"Unsupported model: {model_name}")


def _candidate_dirs(baseline_subdir, default_dir, checkpoint_root):
    dirs = []
    if checkpoint_root:
        root = Path(checkpoint_root)
        dirs.append(root / baseline_subdir)
        dirs.append(root)
    dirs.append(default_dir)
    return dirs


# ---------------------------------------------------------------------------
# Dynamic imports of baseline code (SparseKGC owns training/inference; we only
# reuse its model classes / Runner for scoring already-trained checkpoints)
# ---------------------------------------------------------------------------

def _ensure_sys_path(path):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


_trad_models_module = None


def _load_traditional_model_classes():
    global _trad_models_module
    if _trad_models_module is None:
        _ensure_sys_path(TRADITIONAL_DIR)
        _trad_models_module = importlib.import_module("models.kge_models")
    return _trad_models_module


_hogrn_run_module = None


def _load_hogrn_run_module():
    global _hogrn_run_module
    if _hogrn_run_module is None:
        _ensure_sys_path(HOGRN_DIR)
        spec = importlib.util.spec_from_file_location("hogrn_run_module", HOGRN_DIR / "run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _hogrn_run_module = module
    return _hogrn_run_module


# ---------------------------------------------------------------------------
# Model adapters: each exposes score_batch(query_h_bsr, query_r_bsr) -> raw
# scores as an (batch, num_entities) ndarray with columns in BSR entity order.
# ---------------------------------------------------------------------------

class TraditionalAdapter:
    """TransE / ConvE / TuckER from baselines/tranditional.

    KGData builds ent2id/rel2id the exact same way as the BSR-compatible
    mapping (sorted entities, sorted base relations + reverse offset), so the
    model's internal id space already equals the BSR id space -- no
    translation needed.
    """

    TRANSE_MARGIN = 9.0  # matches baselines/tranditional/run_all.py training config

    def __init__(self, model_name, checkpoint_path, num_entities, num_base_rel, device):
        models_module = _load_traditional_model_classes()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        emb_dim = int(state_dict["ent_emb.weight"].shape[1])
        margin = self.TRANSE_MARGIN if model_name == "TransE" else 0.0

        args = argparse.Namespace(
            num_ent=num_entities,
            num_rel=num_base_rel * 2,
            emb_dim=emb_dim,
            margin=margin,
        )
        model_cls = {"TransE": models_module.TransE,
                     "ConvE": models_module.ConvE,
                     "TuckER": models_module.TuckER}[model_name]
        self.model = model_cls(args)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.emb_dim = emb_dim
        self.margin = margin

    def describe_config(self):
        bits = [f"checkpoint={self.checkpoint_path.name}", f"emb_dim={self.emb_dim}"]
        if self.margin:
            bits.append(f"margin={self.margin}")
        return "; ".join(bits)

    @torch.no_grad()
    def score_batch(self, query_h_bsr, query_r_bsr):
        h = torch.as_tensor(query_h_bsr, dtype=torch.long, device=self.device)
        r = torch.as_tensor(query_r_bsr, dtype=torch.long, device=self.device)
        scores = self.model(h, r)
        return scores.detach().cpu().numpy()

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Defaults mirroring baselines/HoGRN/run.py's argparse, used as a base before
# overriding with the hyperparameters captured inside the checkpoint itself.
HOGRN_PARAM_DEFAULTS = dict(
    name="bsr_export", dataset="FB15K-237-10", model="hogrn", score_func="conve", opn="mult",
    batch_size=128, max_epochs=9999, gamma=40.0, gpu="0",
    l2=0.0, lr=0.001, lbl_smooth=0.1, num_workers=0, seed=41504,
    restore=False, bias=False, rel_reason=False, pre_reason=False,
    reason_type="mixdrop", act_type="tanh", rel_norm=False,
    init_dim=100, gcn_dim=100, embed_dim=100, gcn_layer=1, dropout=0.0, hid_drop=0.0,
    relmix_dim=200, chamix_dim=200, rel_mask=0.0, chan_drop=0.0, edge_drop=0.0,
    temperature=1.0, sim_decay=0.0, rel_drop=0.0,
    hid_drop2=0.3, feat_drop=0.3, k_w=10, k_h=10, num_filt=32, ker_sz=3,
    log_dir=str(HOGRN_DIR / "log"), config_dir=str(HOGRN_DIR / "config"),
    data_dir=str(REPO_ROOT / "datasets"),
    dump_errors=False, topk=10,
)

# Keys that describe the *run context* rather than the trained architecture --
# these must come from our export invocation, not from the checkpoint's saved args.
_HOGRN_INFERENCE_CONTEXT_KEYS = {
    "dataset", "gpu", "dump_errors", "topk", "name", "log_dir", "config_dir",
    "data_dir", "restore", "max_epochs", "score_func",
}


class HoGRNAdapter:
    """Reuses baselines/HoGRN/run.py's Runner to load the trained model and
    score all entities, then remaps HoGRN's internal id space (insertion-order
    + lower-cased strings) onto the BSR-compatible id space.
    """

    def __init__(self, dataset_name, score_func, checkpoint_path, data_root, gpu,
                 entities, base_relations, device):
        run_module = _load_hogrn_run_module()
        saved = torch.load(checkpoint_path, map_location="cpu")
        saved_args = saved.get("args", {}) or {}

        params = argparse.Namespace(**HOGRN_PARAM_DEFAULTS)
        for key, value in saved_args.items():
            if key not in _HOGRN_INFERENCE_CONTEXT_KEYS:
                setattr(params, key, value)

        params.dataset = dataset_name
        params.score_func = score_func
        params.gpu = str(gpu) if gpu >= 0 else "-1"
        params.data_dir = str(data_root)
        params.name = f"bsr_export_{dataset_name}_{score_func}"
        params.restore = False
        params.dump_errors = False
        params.log_dir = str(HOGRN_DIR / "log")
        params.config_dir = str(HOGRN_DIR / "config")
        params.num_workers = 0

        self.runner = run_module.Runner(params)
        self.runner.load_model(str(checkpoint_path))
        self.runner.model.eval()
        self.device = self.runner.device
        self.checkpoint_path = checkpoint_path
        self.score_func = score_func
        self.params = params

        # HoGRN lower-cases entity/relation strings; verified (see exploration)
        # that lower-casing causes zero collisions for every dataset in this repo.
        self.ent_bsr_to_internal = np.array(
            [self.runner.ent2id[e.lower()] for e in entities], dtype=np.int64)
        self.rel_base_bsr_to_internal = np.array(
            [self.runner.rel2id[r.lower()] for r in base_relations], dtype=np.int64)
        self.internal_num_base_rel = self.runner.p.num_rel
        self.num_base_rel = len(base_relations)

    def describe_config(self):
        return ("checkpoint={}; score_func={}; embed_dim={}; gcn_dim={}; opn={}".format(
            self.checkpoint_path.name, self.score_func,
            self.params.embed_dim, self.params.gcn_dim, self.params.opn))

    def _to_internal_relation(self, query_r_bsr):
        is_reverse = query_r_bsr >= self.num_base_rel
        base_bsr = np.where(is_reverse, query_r_bsr - self.num_base_rel, query_r_bsr)
        base_internal = self.rel_base_bsr_to_internal[base_bsr]
        return np.where(is_reverse, base_internal + self.internal_num_base_rel, base_internal)

    @torch.no_grad()
    def score_batch(self, query_h_bsr, query_r_bsr):
        h_internal = self.ent_bsr_to_internal[query_h_bsr]
        r_internal = self._to_internal_relation(query_r_bsr)
        h = torch.as_tensor(h_internal, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r_internal, dtype=torch.long, device=self.device)
        scores, _ = self.runner.model.forward(h, r)
        scores = scores.detach().cpu().numpy()
        # Reorder columns from HoGRN-internal entity order to BSR entity order.
        return scores[:, self.ent_bsr_to_internal]

    def close(self):
        del self.runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_adapter(model_name, dataset_name, score_func, checkpoint_path,
                  entities, base_relations, args, device):
    if model_name in TRAD_MODELS:
        return TraditionalAdapter(model_name, checkpoint_path, len(entities), len(base_relations), device)
    if model_name in HOGRN_MODELS:
        return HoGRNAdapter(dataset_name, score_func, checkpoint_path, args.data_root, args.gpu,
                            entities, base_relations, device)
    raise ValueError(f"Unsupported model: {model_name}")


# ---------------------------------------------------------------------------
# Per-(model, dataset, split) export
# ---------------------------------------------------------------------------

def build_queries(triples_bsr, num_base_rel):
    """
    For original triple i = (h, r, t):
      query_index 2*i   = tail prediction:  (query_h, query_r, query_t) = (h, r, t),       gold = t
      query_index 2*i+1 = head prediction:  (query_h, query_r, query_t) = (t, r+num_base_rel, h), gold = h
    """
    queries = []
    for i, (h, r, t) in enumerate(triples_bsr):
        queries.append((2 * i, "tail", h, r, t, h, r, t, t))
        queries.append((2 * i + 1, "head", h, r, t, t, r + num_base_rel, h, h))
    return queries


def export_split(model_name, dataset_name, split, adapter, entities,
                 triples_bsr, true_obj_filter, num_base_rel, top_k,
                 summary_writer, candidates_writer):
    queries = build_queries(triples_bsr, num_base_rel)
    n = len(queries)

    reciprocal_rank_sum = 0.0
    hits = {1: 0, 3: 0, 10: 0}

    for start in range(0, n, EVAL_BATCH_SIZE):
        batch = queries[start:start + EVAL_BATCH_SIZE]
        qh = np.asarray([q[5] for q in batch], dtype=np.int64)
        qr = np.asarray([q[6] for q in batch], dtype=np.int64)
        scores = adapter.score_batch(qh, qr)  # (batch, num_ent), raw scores in BSR column order

        for row, (qi, direction, oh, orr, ot, h_, r_, t_, gold) in enumerate(batch):
            raw_row = scores[row]
            gold_score = float(raw_row[gold])

            true_objects = true_obj_filter.get((h_, r_))
            if true_objects:
                mask_idx = [e for e in true_objects if e != gold]
            else:
                mask_idx = None
            if mask_idx:
                filtered_row = raw_row.copy()
                filtered_row[mask_idx] = -1e7
            else:
                filtered_row = raw_row

            target = filtered_row[gold]
            greater = int(np.sum(filtered_row > target))
            equal = int(np.sum(filtered_row == target))
            filtered_rank = greater + (equal + 1) / 2.0

            top1_idx = int(np.argmax(filtered_row))
            top1_score = float(filtered_row[top1_idx])

            reciprocal_rank_sum += 1.0 / filtered_rank
            for k in (1, 3, 10):
                if filtered_rank <= k:
                    hits[k] += 1

            summary_writer.writerow([
                model_name, dataset_name, split, qi, direction,
                oh, orr, ot,
                h_, r_, t_, gold,
                f"{gold_score:.10g}", f"{filtered_rank:.6f}",
                entities[top1_idx], f"{top1_score:.10g}",
            ])

            k_eff = min(top_k, raw_row.shape[0])
            top_idx = np.argpartition(-raw_row, k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-raw_row[top_idx])]
            for rank_pos, cand_idx in enumerate(top_idx.tolist(), start=1):
                candidates_writer.writerow([
                    model_name, dataset_name, split, qi, direction,
                    oh, orr, ot,
                    h_, r_, t_, gold,
                    rank_pos, entities[cand_idx], f"{float(raw_row[cand_idx]):.10g}",
                ])

    return {
        "mrr": reciprocal_rank_sum / n,
        "hits@1": hits[1] / n,
        "hits@3": hits[3] / n,
        "hits@10": hits[10] / n,
        "num_queries": n,
    }


# ---------------------------------------------------------------------------
# Manifest / README / SCHEMA
# ---------------------------------------------------------------------------

def git_commit_hash():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_remote_url():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "config", "--get", "remote.origin.url"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_readme(path, args, source_repo, source_commit, export_time):
    path.write_text(
        "# BSR Routing/Hybrid-Expert Prediction Export\n\n"
        "This directory contains prediction artifacts exported from SparseKGC\n"
        "checkpoints so a separate Case-Path BSR repo can use HoGRN, TransE,\n"
        "ConvE and TuckER as routing / hybrid experts. SparseKGC owns training,\n"
        "checkpoint loading, inference and CSV export -- the BSR repo only\n"
        "consumes these CSV artifacts; no training code is duplicated there.\n\n"
        "## Provenance\n\n"
        f"- Source repo: {source_repo}\n"
        f"- Source commit: {source_commit}\n"
        f"- Export time (UTC): {export_time}\n"
        f"- Models: {', '.join(args.models)}\n"
        f"- Datasets: {', '.join(args.datasets)}\n"
        f"- Splits: {', '.join(args.splits)}\n"
        f"- Seed: {args.seed}\n"
        f"- Top-K export: {args.top_k}\n\n"
        "## Evaluation protocol\n\n"
        f"- {PROTOCOL_LABEL}\n"
        "- For every original (h, r, t) triple in a split, two queries are produced "
        "in this order: a tail-prediction query (query_h=h, query_r=r, query_t=t, "
        "gold_entity=t), then a head-prediction query answered via the reverse "
        "relation (query_h=t, query_r=r+num_base_rel, query_t=h, gold_entity=h). "
        "query_index runs 0, 1, 2, ... following this tail-then-head interleaving "
        "in original triple order.\n"
        "- Filtered ranking uses all true answers from train+valid+test, masking "
        "every other true answer to a very low score while preserving the gold "
        "entity's own raw score, then computing a tie-aware rank: "
        "filtered_rank = num_scores_greater_than_gold + (num_scores_equal_to_gold + 1) / 2.\n\n"
        "## ID mapping\n\n"
        f"- entity_id: {ENTITY_MAPPING_LABEL}\n"
        f"- base_relation_id: sorted(all base relations from train+valid+test).index\n"
        "- reverse_relation_id = base_relation_id + num_base_relations\n"
        "- All *_h/*_r/*_t/gold_entity fields in the CSVs are these BSR-compatible "
        "integer ids; entity *string* fields (top1_entity, candidate_entity) are "
        "the raw entity strings from the dataset files (sorted(entities)[id]).\n\n"
        "## Layout\n\n"
        "```\n"
        "external_predictions/\n"
        "  README.md\n"
        "  SCHEMA.md\n"
        "  prediction_manifest.csv\n"
        "  valid_predictions/\n"
        "    prediction_manifest.csv\n"
        "    {MODEL}/{DATASET}/valid_query_summary.csv\n"
        "    {MODEL}/{DATASET}/valid_topk_candidates.csv\n"
        "  test_predictions/\n"
        "    prediction_manifest.csv\n"
        "    {MODEL}/{DATASET}/test_query_summary.csv\n"
        "    {MODEL}/{DATASET}/test_topk_candidates.csv\n"
        "```\n\n"
        "See SCHEMA.md for column-by-column documentation of every CSV.\n",
        encoding="utf-8",
    )


def write_schema(path):
    path.write_text(
        "# CSV Schema Reference\n\n"
        "## query_summary.csv\n\n"
        "One row per query (two queries per original triple: tail then head).\n\n"
        "| column | meaning |\n"
        "| --- | --- |\n"
        "| model | exporting model name (HoGRN / TransE / ConvE / TuckER) |\n"
        "| dataset | dataset name |\n"
        "| split | valid / test |\n"
        "| query_index | 0-based index; for original triple i: 2*i = tail query, 2*i+1 = head query |\n"
        "| direction | 'tail' or 'head' |\n"
        "| original_h / original_r / original_t | the original (h, r, t) triple, as BSR-compatible integer ids |\n"
        "| query_h / query_r / query_t | the query actually scored: tail query = (h, r, t); "
        "head query = (t, r+num_base_rel, h), as BSR-compatible integer ids |\n"
        "| gold_entity | the correct answer entity id for this query (BSR-compatible integer id) |\n"
        "| gold_score | the model's raw score for gold_entity |\n"
        "| filtered_rank | tie-aware filtered rank of gold_entity: "
        "num_scores_greater_than_gold + (num_scores_equal_to_gold + 1) / 2, "
        "computed after masking every other true answer (from train+valid+test) to -1e7 "
        "while keeping the gold entity's own raw score |\n"
        "| top1_entity | raw entity string of the top-ranked entity under the filtered scoring |\n"
        "| top1_score | filtered score of the top-ranked entity |\n\n"
        "## topk_candidates.csv\n\n"
        "Top-K entities by **raw, unfiltered** model score for each query "
        "(K = --top-k, default 200; truncated to num_entities when smaller).\n\n"
        "| column | meaning |\n"
        "| --- | --- |\n"
        "| model / dataset / split / query_index / direction | same as query_summary |\n"
        "| original_h / original_r / original_t | original triple, BSR-compatible integer ids |\n"
        "| query_h / query_r / query_t | query actually scored, BSR-compatible integer ids |\n"
        "| gold_entity | correct answer entity id, BSR-compatible integer id |\n"
        "| candidate_rank | 1-based rank of this candidate by raw model score (descending) |\n"
        "| candidate_entity | raw entity string (sorted(entities)[candidate id]) |\n"
        "| candidate_score | raw model score for this candidate |\n\n"
        "## prediction_manifest.csv\n\n"
        "One row per (model, dataset, split). The root-level manifest's "
        "summary_file/candidates_file paths are relative to the export root; "
        "each split-level manifest's paths are relative to that split directory.\n\n"
        "| column | meaning |\n"
        "| --- | --- |\n"
        "| model / dataset / split | identifies the export run |\n"
        "| mrr / hits@1 / hits@3 / hits@10 | bidirectional filtered metrics, "
        "averaged over all queries (tail and head), recomputable from query_summary.filtered_rank |\n"
        "| num_queries | number of rows in the corresponding query_summary.csv "
        "(= 2 * number of triples in the split) |\n"
        "| summary_file / candidates_file | relative paths to the CSV artifacts |\n"
        "| source_repo | SparseKGC git remote URL |\n"
        "| source_commit | SparseKGC git commit hash used for this export |\n"
        "| source_config | checkpoint filename + key model hyperparameters used for inference |\n"
        "| seed | random seed used for the export run |\n"
        "| protocol | evaluation protocol label |\n"
        "| entity_mapping | how entity_id is derived |\n"
        "| relation_mapping | how relation_id is derived |\n"
        "| top_k_export | K used for topk_candidates.csv |\n"
        "| export_time_utc | ISO-8601 UTC timestamp of the export run |\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_export(output_root, manifest_records, dataset_info, top_k):
    print("\n" + "=" * 72)
    print("Validating export...")
    print("=" * 72)
    errors = []

    for rec in manifest_records:
        model, dataset, split = rec["model"], rec["dataset"], rec["split"]
        summary_path = output_root / rec["summary_file_root"]
        candidates_path = output_root / rec["candidates_file_root"]
        label = f"{model}/{dataset}/{split}"

        for rel_path in (rec["summary_file_root"], rec["candidates_file_root"],
                         rec["summary_file_split"], rec["candidates_file_split"]):
            if Path(rel_path).is_absolute():
                errors.append(f"[{label}] manifest path is absolute: {rel_path}")

        if not summary_path.exists():
            errors.append(f"[{label}] missing summary file: {summary_path}")
            continue
        if not candidates_path.exists():
            errors.append(f"[{label}] missing candidates file: {candidates_path}")
            continue

        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ranks = []
            for row in reader:
                ranks.append(float(row["filtered_rank"]))

        num_triples = dataset_info[dataset]["num_triples"][split]
        expected_queries = 2 * num_triples
        if len(ranks) != expected_queries:
            errors.append(f"[{label}] query_summary row count {len(ranks)} != 2*num_triples {expected_queries}")
        if int(rec["num_queries"]) != len(ranks):
            errors.append(f"[{label}] manifest num_queries {rec['num_queries']} != query_summary row count {len(ranks)}")

        if ranks:
            recomputed = {
                "mrr": sum(1.0 / r for r in ranks) / len(ranks),
                "hits@1": sum(1 for r in ranks if r <= 1) / len(ranks),
                "hits@3": sum(1 for r in ranks if r <= 3) / len(ranks),
                "hits@10": sum(1 for r in ranks if r <= 10) / len(ranks),
            }
            for key in ("mrr", "hits@1", "hits@3", "hits@10"):
                if abs(recomputed[key] - float(rec[key])) > 1e-4:
                    errors.append(f"[{label}] manifest {key}={rec[key]} != recomputed {recomputed[key]:.5f}")

        with open(candidates_path, encoding="utf-8") as f:
            num_candidate_rows = sum(1 for _ in f) - 1  # minus header
        num_entities = dataset_info[dataset]["num_entities"]
        expected_candidate_rows = expected_queries * min(top_k, num_entities)
        if num_candidate_rows != expected_candidate_rows:
            errors.append(
                f"[{label}] topk_candidates row count {num_candidate_rows} != "
                f"num_queries*min(top_k,num_entities) {expected_candidate_rows}"
            )

        print(f"[{label}] mrr={rec['mrr']} hits@1={rec['hits@1']} hits@3={rec['hits@3']} hits@10={rec['hits@10']}")

    print("=" * 72)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"All validation checks passed for {len(manifest_records)} (model, dataset, split) export(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        (output_root / f"{split}_predictions").mkdir(parents=True, exist_ok=True)

    source_repo = git_remote_url()
    source_commit = git_commit_hash()

    manifest_rows_root = []
    manifest_rows_by_split = {split: [] for split in args.splits}
    manifest_records = []
    dataset_info = {}

    for dataset_name in args.datasets:
        dataset_dir = data_root / dataset_name
        if not dataset_dir.exists():
            print(f"[skip dataset] {dataset_name}: path not found at {dataset_dir}")
            continue

        print(f"\n=== Dataset: {dataset_name} ===")
        raw_triples = read_raw_triples(dataset_dir)
        entities, ent2id, base_relations, rel2id = build_bsr_mapping(raw_triples)
        num_base_rel = len(base_relations)
        translated = translate_triples(raw_triples, ent2id, rel2id)
        true_obj_filter = build_true_object_filter(translated, num_base_rel)
        dataset_info[dataset_name] = {
            "num_entities": len(entities),
            "num_triples": {split: len(translated[split]) for split in ("train", "valid", "test")},
        }
        print(f"  num_entities={len(entities)} num_base_relations={num_base_rel} "
              f"train/valid/test={len(translated['train'])}/{len(translated['valid'])}/{len(translated['test'])}")

        for model_name in args.models:
            checkpoint_path, score_func = resolve_checkpoint(model_name, dataset_name, args.checkpoint_root)
            if checkpoint_path is None:
                print(f"  [skip] {model_name}: no checkpoint found for dataset {dataset_name}")
                continue

            print(f"  [load] {model_name}: checkpoint={checkpoint_path}")
            torch.manual_seed(args.seed)
            adapter = build_adapter(model_name, dataset_name, score_func, checkpoint_path,
                                    entities, base_relations, args, device)
            try:
                for split in args.splits:
                    triples_bsr = translated[split]
                    split_dir = output_root / f"{split}_predictions" / model_name / dataset_name
                    summary_path = split_dir / f"{split}_query_summary.csv"
                    candidates_path = split_dir / f"{split}_topk_candidates.csv"
                    split_dir.mkdir(parents=True, exist_ok=True)

                    t0 = __import__("time").perf_counter()
                    with open(summary_path, "w", newline="", encoding="utf-8") as sf, \
                            open(candidates_path, "w", newline="", encoding="utf-8") as cf:
                        sw = csv.writer(sf)
                        sw.writerow(QUERY_SUMMARY_HEADER)
                        cw = csv.writer(cf)
                        cw.writerow(TOPK_HEADER)
                        metrics = export_split(model_name, dataset_name, split, adapter, entities,
                                               triples_bsr, true_obj_filter, num_base_rel, args.top_k,
                                               sw, cw)
                    elapsed = __import__("time").perf_counter() - t0

                    summary_root_rel = summary_path.relative_to(output_root)
                    candidates_root_rel = candidates_path.relative_to(output_root)
                    split_base = output_root / f"{split}_predictions"
                    summary_split_rel = summary_path.relative_to(split_base)
                    candidates_split_rel = candidates_path.relative_to(split_base)

                    export_time = datetime.now(timezone.utc).isoformat()
                    common = [
                        model_name, dataset_name, split,
                        f"{metrics['mrr']:.5f}", f"{metrics['hits@1']:.5f}",
                        f"{metrics['hits@3']:.5f}", f"{metrics['hits@10']:.5f}",
                        metrics["num_queries"],
                    ]
                    tail = [
                        source_repo, source_commit, adapter.describe_config(), args.seed,
                        PROTOCOL_LABEL, ENTITY_MAPPING_LABEL, RELATION_MAPPING_LABEL,
                        args.top_k, export_time,
                    ]
                    manifest_rows_root.append(common + [str(summary_root_rel), str(candidates_root_rel)] + tail)
                    manifest_rows_by_split[split].append(
                        common + [str(summary_split_rel), str(candidates_split_rel)] + tail)
                    manifest_records.append({
                        "model": model_name, "dataset": dataset_name, "split": split,
                        "mrr": f"{metrics['mrr']:.5f}", "hits@1": f"{metrics['hits@1']:.5f}",
                        "hits@3": f"{metrics['hits@3']:.5f}", "hits@10": f"{metrics['hits@10']:.5f}",
                        "num_queries": metrics["num_queries"],
                        "summary_file_root": str(summary_root_rel), "candidates_file_root": str(candidates_root_rel),
                        "summary_file_split": str(summary_split_rel), "candidates_file_split": str(candidates_split_rel),
                    })
                    print(f"    [{split}] mrr={metrics['mrr']:.5f} hits@1={metrics['hits@1']:.5f} "
                          f"hits@3={metrics['hits@3']:.5f} hits@10={metrics['hits@10']:.5f} "
                          f"num_queries={metrics['num_queries']} ({elapsed:.1f}s)")
            finally:
                adapter.close()

    # Reorder manifest columns: header is
    # model,dataset,split,mrr,...,num_queries,summary_file,candidates_file,source_repo,...
    # but we appended [common(8) + [summary,candidates] + tail(9)] = 19 columns matching MANIFEST_HEADER.
    write_csv(output_root / "prediction_manifest.csv", MANIFEST_HEADER, manifest_rows_root)
    for split in args.splits:
        write_csv(output_root / f"{split}_predictions" / "prediction_manifest.csv",
                  MANIFEST_HEADER, manifest_rows_by_split[split])

    export_time = datetime.now(timezone.utc).isoformat()
    write_readme(output_root / "README.md", args, source_repo, source_commit, export_time)
    write_schema(output_root / "SCHEMA.md")

    if not manifest_records:
        print("\nNo (model, dataset, split) combinations were exported -- nothing to validate.")
        return

    validate_export(output_root, manifest_records, dataset_info, args.top_k)

    exported_models = sorted({rec["model"] for rec in manifest_records})
    print("\nModels confirmed usable as BSR routing experts (predictions exported & validated): "
          + ", ".join(exported_models))
    missing = [m for m in ["HoGRN", "TransE", "ConvE", "TuckER"] if m not in exported_models]
    if missing:
        print("Not exported (no checkpoint found for the requested datasets): " + ", ".join(missing))


if __name__ == "__main__":
    main()
