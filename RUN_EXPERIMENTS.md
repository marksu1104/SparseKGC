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

From the parent directory, submit one baseline per job:

```bash
sbatch exp_traditional.sh
sbatch exp_hogrn.sh
sbatch exp_dackgr.sh
sbatch exp_probcbr.sh
```

Each script has a fixed Slurm job name and writes one main log directly under `SparseKGC/outputs/`.

The Slurm script uses the `.venv` project virtual environment.

## Job Logs

Slurm bootstrap logs, main per-baseline job logs, metrics CSV files, and per-model/per-dataset training logs are written under:

```text
outputs/
```

Main job logs are named by baseline and job id, for example:

```text
outputs/traditional_151700.log
outputs/hogrn_151701.log
outputs/dackgr_151702.log
outputs/probcbr_151703.log
```

Training logs are grouped by baseline:

```text
outputs/traditional/TransE_WD-singer.log
outputs/hogrn/HoGRN_conve_WD-singer.log
outputs/dackgr/DacKGR_train_infer_WD-singer.log
outputs/probcbr/Prob-CBR_WD-singer.log
```

## Official Outputs

Each baseline writes exactly one official metrics CSV:

| Baseline | Metrics CSV |
| --- | --- |
| Traditional | `outputs/traditional_metrics.csv` |
| HoGRN | `outputs/hogrn_metrics.csv` |
| DacKGR | `outputs/dackgr_metrics.csv` |
| Prob-CBR | `outputs/probcbr_metrics.csv` |

All official CSV files use this schema:

```text
Dataset,Model,MRR,Hits@1,Hits@3,Hits@10,seconds
```

Final metric logs use:

```text
FINAL_EVAL_METRICS baseline=<baseline> model=<model> dataset=<dataset> split=holdout mrr=<...> h1=<...> h3=<...> h10=<...>
```

## Model Scope

- Traditional keeps the standard models: `TransE`, `DistMult`, `ComplEx`, `ConvE`, `TuckER`, `RotatE`.
- HoGRN runs `conve` only.
- DacKGR runs the full README-style pipeline: `process_data` -> `pretrain_conve` -> `point.rs.conve` train plus final inference. Only `point.rs.conve` final inference is written to `outputs/dackgr_metrics.csv`.
- Prob-CBR has one model label: `Prob-CBR`.

## Cleanup Policy

The official path is `run_baseline.py` plus each baseline's native runner. Older summaries and full-matrix timing files are historical artifacts, not official outputs for new runs.
