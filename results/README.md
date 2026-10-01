# results/

One CSV per run (`<run_name>.csv`) plus `all_results.jsonl` (every row from every run, one JSON object per line). Each row is appended automatically by `evaluate()` in `train.py` (validation during training, and `--eval_only` runs for `val`, `test` and `3dpw`); columns include split, checkpoint, epoch, PVE, PA-PVE, MPJPE, PA-MPJPE, precision/recall/F1 and `n_seen_in_training`, with empty cells for metrics that were not measured.
These files are the source of truth: metric numbers written as comments in the `*.sh` scripts are historical and superseded by these files.
Directory is set by `train.py --results_dir` (default `results/` in the repo).
