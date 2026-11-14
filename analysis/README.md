# Benchmark analysis

Compiles benchmark CSVs from `../results/` into scores, graphs and summaries per
profile, mapped to the research questions. Everything the notebook renders is
also written to `exports/` (figures as SVG+PNG, tables as CSV+markdown) for the
report.

## Setup

```
git submodule update --init
uv sync
```

Fallback without uv: `python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install pytest`

## Run

```
uv run jupyter lab analysis.ipynb          # interactive
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # regenerate everything
uv run pytest                              # loader + regex tests
```

## Notes

- Every `*_summary.csv` in `../results/` loads regardless of the harness
  `--output-prefix`; `results/archive/` is excluded. `DATE_FILTER` in cell 0
  (default `None`) narrows to one session by the timestamp in the filename.
- capi-autoscaler runs are split into cohorts by source-file date:
  `placementcontrol` for files dated on/after `POST_PLACEMENTCONTROL_FROM`
  (`capi_bench_analysis/loading.py`, currently 20260831), `pre-placementcontrol`
  before that. cas/karpenter runs are `baseline` either way. All sections use
  the placementcontrol cohort plus baselines; pre-placementcontrol rows only
  feed the RQ4 pre-vs-post comparison.
- Every run pools into the averages: no supersession, all files count. Curate
  by archiving files you don't want included, or drop single runs via
  `EXCLUDED_RUN_IDS` in cell 0. Duplicate run_ids across files (prefix copies)
  load once.
- RQ2/RQ3 sections need `*_solver.csv` files, which the harness writes for
  capi-autoscaler runs since the solver-metrics change (`benchmark/solver_logs.py`).
- `INSTANCE_HOURLY_COST` and `PROFILES` are imported from
  `../capi-autoscaler/benchmarks/harness/benchmark/config.py` (the submodule),
  not duplicated here.
