# SparseKGC Experiment Runbook

This project runs four baselines separately on HPC. The single entry point is:

```bash
python run_baseline.py <baseline> [options]
```

Supported baselines:

- `traditional`
- `hogrn`
- `dackgr`
- `probcbr`

## Submitting Jobs

The Slurm submission scripts live one level up, in the parent `marksu/` directory
(not inside this repo), and each activates the project `.venv` before calling
`run_baseline.py`:

```bash
cd /storage/professor/csliao/marksu
sbatch exp_traditional.sh
sbatch exp_hogrn.sh
sbatch exp_dackgr.sh
sbatch exp_probcbr.sh
```

Each script runs all 6 datasets (WD-singer, FB15K-237-10, WN18RR, FB15K-237-20,
FB15K-237-50, NELL23K) by default. To run a subset, override `DATASETS`
(and `MODELS` for `exp_traditional.sh`) via `--export`, e.g.:

```bash
sbatch --export=ALL,DATASETS="WD-singer FB15K-237-10" exp_dackgr.sh
sbatch --export=ALL,MODELS="DistMult ComplEx" exp_traditional.sh
```

For DacKGR, each dataset's 3-stage pipeline (`process_data` -> `pretrain_conve`
-> `train_infer`) takes a long time, so it's normal to chain one job per
dataset on `gpu_long` with `--dependency=afterany:<prev_jobid>` rather than
submitting all datasets in a single job.

## Job Logs

Slurm bootstrap logs (one per submitted job) are written to
`/storage/professor/csliao/marksu/logs/`, named `<baseline>_run<N>_<jobid>.log`.

Metrics CSV files and per-model/per-dataset training logs are written under:

```text
outputs/
```

Training logs are grouped by baseline:

```text
outputs/traditional/TransE_WD-singer.log
outputs/hogrn/HoGRN_conve_WD-singer.log
outputs/dackgr/DacKGR_train_infer_WD-singer.log
outputs/probcbr/Prob-CBR_WD-singer.log
```

## Official Outputs

Each baseline writes exactly one official metrics CSV, upserted by
`(Dataset, Model)` so reruns always keep only the latest row per combo
(see `scripts/metrics_csv.py`):

| Baseline | Metrics CSV |
| --- | --- |
| Traditional | `outputs/traditional_metrics.csv` |
| HoGRN | `outputs/hogrn_metrics.csv` |
| DacKGR | `outputs/dackgr_metrics.csv` |
| Prob-CBR | `outputs/probcbr_metrics.csv` |

All official CSV files use this schema:

```text
Dataset,Model,MRR_Tail,MRR_Head,MRR_Avg,Hits@1_Tail,Hits@1_Head,Hits@1_Avg,Hits@3_Tail,Hits@3_Head,Hits@3_Avg,Hits@10_Tail,Hits@10_Head,Hits@10_Avg,seconds
```

Final metric logs use:

```text
FINAL_EVAL_METRICS baseline=<baseline> model=<model> dataset=<dataset> split=test mrr_tail=<...> mrr_head=<...> mrr_avg=<...> h1_tail=<...> h1_head=<...> h1_avg=<...> h3_tail=<...> h3_head=<...> h3_avg=<...> h10_tail=<...> h10_head=<...> h10_avg=<...>
```

## Model Scope

- Traditional keeps the standard models: `TransE`, `DistMult`, `ComplEx`, `ConvE`, `TuckER`, `RotatE`.
  - DistMult and ComplEx use `l2=0.0` (Adam `weight_decay` on these bilinear
    models collapses the embeddings to a degenerate fixed point); all other
    models use the literature-common settings in `MODEL_CONFIGS`.
- HoGRN runs `conve` only.
- DacKGR runs the full README-style pipeline: `process_data` -> `pretrain_conve` -> `point.rs.conve` train plus final inference. Only `point.rs.conve` final inference is written to `outputs/dackgr_metrics.csv` (Model = `point.rs.conve`).
- Prob-CBR has one model label: `Prob-CBR`.

## Cleanup Policy

The official path is `run_baseline.py` plus each baseline's native runner. Older summaries and full-matrix timing files are historical artifacts, not official outputs for new runs.
