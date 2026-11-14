"""Load benchmark CSVs, pooling every run present in results/."""

import csv
import importlib.util
import re
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = BENCHMARKS_ROOT / "results"
HARNESS_CONFIG_PATH = BENCHMARKS_ROOT / "capi-autoscaler" / "benchmarks" / "harness" / "benchmark" / "config.py"

# Invalidated data lives in results/archive/, so no date filter is needed;
# set one to narrow a session (e.g. "20260830").
DEFAULT_DATE_FILTER = None

# capi-autoscaler runs from files dated on/after this are the placementcontrol
# cohort; earlier capi files are pre-placementcontrol. cas/karpenter runs are
# baselines either way (placement control only changes capi-autoscaler).
POST_PLACEMENTCONTROL_FROM = "20260831"

# Any harness --output-prefix, then _YYYYMMDD_HHMMSS_<kind>.csv
BATCH_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<stamp>\d{8}_\d{6})_(?P<kind>summary|timeseries|solver)\.csv$")

NUMERIC_COLUMNS = [
    "latency_s",
    "first_unschedulable_s",
    "all_scheduled_s",
    "all_running_s",
    "worker_nodes",
    "workload_cpu_util_pct",
    "workload_mem_util_pct",
    "cost_per_hour",
]

SOLVER_COLUMNS = [
    "autoscaler", "scenario", "run_id", "attempt", "status",
    "objective", "unplaced", "used_beam", "solve_time_secs",
]


def load_harness_config():
    """Import benchmarks/harness/benchmark/config.py by path, so
    INSTANCE_HOURLY_COST and PROFILES have a single source of truth."""
    spec = importlib.util.spec_from_file_location("bench_config", HARNESS_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_batches(results_dir=RESULTS_DIR, kind="summary",
                 date_filter=DEFAULT_DATE_FILTER):
    """All matching CSVs regardless of harness --output-prefix, sorted by the
    timestamp token (then prefix) so supersession stays chronological."""
    matches = []
    for f in results_dir.glob(f"*_{kind}.csv"):
        m = BATCH_NAME_RE.match(f.name)
        if not m or m.group("kind") != kind:
            continue
        if date_filter and not m.group("stamp").startswith(date_filter):
            continue
        matches.append((m.group("stamp"), m.group("prefix"), f))
    return [f for _, _, f in sorted(matches)]


def cohort_for(autoscaler: str, file_date: str) -> str:
    if autoscaler != "capi-autoscaler":
        return "baseline"
    if file_date < POST_PLACEMENTCONTROL_FROM:
        return "pre-placementcontrol"
    return "placementcontrol"


def load_summaries(results_dir=RESULTS_DIR,
                   date_filter=DEFAULT_DATE_FILTER) -> pd.DataFrame:
    """Pool every run from every file in results/: means are over everything
    present, so curation is by archiving files (plus the notebook's
    EXCLUDED_RUN_IDS for single rows). Duplicate run_ids from prefix-copy
    files load once, first file wins."""
    all_rows = []
    seen_runs: set[tuple[str, str]] = set()
    for f in list_batches(results_dir, "summary", date_filter):
        file_date = BATCH_NAME_RE.match(f.name).group("stamp")[:8]
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                run_key = (row["autoscaler"], row["run_id"])
                if run_key in seen_runs:
                    continue
                seen_runs.add(run_key)
                row["cohort"] = cohort_for(row["autoscaler"], file_date)
                row["source_file"] = f.name
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(columns=["autoscaler", "scenario", "profile", "pod_count",
                                     "run_id", "timed_out", "cohort", "source_file",
                                     *NUMERIC_COLUMNS])
    df = pd.DataFrame(all_rows)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pod_count"] = pd.to_numeric(df["pod_count"], errors="coerce").astype("Int64")
    df["timed_out"] = df["timed_out"] == "True"
    return df.reset_index(drop=True)


def parse_instance_counts(s: str) -> dict[str, int]:
    """'c5.2xlarge=1;m5.4xlarge=8' -> {'c5.2xlarge': 1, 'm5.4xlarge': 8}"""
    if not s or pd.isna(s):
        return {}
    out = {}
    for part in str(s).split(";"):
        if "=" in part:
            t, n = part.split("=", 1)
            out[t] = int(n)
    return out


def explode_instance_counts(summary_df: pd.DataFrame,
                            hourly_cost: dict[str, float]) -> pd.DataFrame:
    """One row per (run, instance type) with count and hourly cost."""
    rows = []
    for _, r in summary_df.iterrows():
        for instance_type, count in parse_instance_counts(r.get("instance_counts")).items():
            rows.append({
                "autoscaler": r["autoscaler"],
                "scenario": r["scenario"],
                "run_id": r["run_id"],
                "instance_type": instance_type,
                "count": count,
                "hourly_cost": count * hourly_cost.get(instance_type, 0.0),
            })
    return pd.DataFrame(rows, columns=["autoscaler", "scenario", "run_id",
                                       "instance_type", "count", "hourly_cost"])


def load_timeseries(surviving_run_ids, results_dir=RESULTS_DIR,
                    date_filter=DEFAULT_DATE_FILTER) -> pd.DataFrame:
    files = list_batches(results_dir, "timeseries", date_filter)
    if not files:
        return pd.DataFrame()
    # Header-only files (a batch mid-write) would upcast every column to object
    frames = [df for df in (pd.read_csv(f) for f in files) if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["run_id"].isin(set(surviving_run_ids))]
    # Prefix-duplicate files repeat the same run_ids; keep each sample once
    return df.drop_duplicates().reset_index(drop=True)


def steady_points(ts_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """First sample per run with every pod running; falls back to the last
    sample for runs that never got there (timeouts). The run's final sample
    can catch Karpenter mid-consolidation with nodes tainted out of the
    worker count, so it must not be used for utilisation or cost."""
    counts = summary_df.set_index("run_id")["pod_count"]
    systems = summary_df.set_index("run_id")["autoscaler"]
    rows = []
    for run_id, g in ts_df.groupby("run_id"):
        g = g.sort_values("elapsed_s")
        # karpenter-consolidation runs hold the full window on purpose; the
        # post-consolidation last sample is the state we want
        if systems.get(run_id) == "karpenter-consolidation":
            rows.append(g.iloc[-1])
            continue
        target = counts.get(run_id)
        steady = g[g["running_pods"] >= target] if target is not None else g.iloc[0:0]
        if len(steady):
            # Karpenter can already be consolidating (nodes tainted out of the
            # worker count) at the first all-running sample; take the one that
            # saw the most allocatable
            rows.append(steady.loc[steady["worker_cpu_allocatable_m"].idxmax()])
        else:
            rows.append(g.iloc[-1])
    if not rows:
        return pd.DataFrame(columns=ts_df.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def settle_times(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Per run: elapsed_s of the last change in worker_nodes, i.e. when the
    fleet reached its final size. 0.0 for runs that never scaled."""
    rows = []
    for run_id, g in ts_df.groupby("run_id"):
        g = g.sort_values("elapsed_s")
        changed = g[g["worker_nodes"].diff().fillna(0) != 0]
        rows.append({"run_id": run_id,
                     "settled_s": float(changed["elapsed_s"].max()) if len(changed) else 0.0})
    return pd.DataFrame(rows, columns=["run_id", "settled_s"])


def recompute_utilisation(summary_df: pd.DataFrame, ts_df: pd.DataFrame) -> pd.DataFrame:
    """Override the summary's utilisation and node-count columns with values
    from each run's steady point. Rows without timeseries data keep their
    recorded values."""
    if ts_df.empty or summary_df.empty:
        return summary_df
    import numpy as np

    sp = steady_points(ts_df, summary_df).set_index("run_id")
    df = summary_df.set_index("run_id")
    cpu = sp["workload_cpu_requested_m"] / sp["worker_cpu_allocatable_m"].replace(0, np.nan) * 100
    mem = sp["workload_mem_requested_mi"] / sp["worker_mem_allocatable_mi"].replace(0, np.nan) * 100
    idx = sp.index.intersection(df.index)
    df.loc[idx, "workload_cpu_util_pct"] = cpu[idx].astype(float).round(1)
    df.loc[idx, "workload_mem_util_pct"] = mem[idx].astype(float).round(1)
    df.loc[idx, "worker_nodes"] = sp.loc[idx, "worker_nodes"]
    return df.reset_index()


def load_solver_metrics(surviving_run_ids=None, results_dir=RESULTS_DIR,
                        date_filter=DEFAULT_DATE_FILTER) -> pd.DataFrame:
    """Solver CSVs are only written by runs after the solver-metrics harness
    change; returns an empty typed frame until such data exists."""
    files = list_batches(results_dir, "solver", date_filter)
    if not files:
        return pd.DataFrame(columns=SOLVER_COLUMNS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if surviving_run_ids is not None:
        df = df[df["run_id"].isin(set(surviving_run_ids))]
    df["used_beam"] = df["used_beam"].astype(str).str.lower() == "true"
    # Prefix-duplicate files repeat the same run_ids; keep each solve once
    return df.drop_duplicates().reset_index(drop=True)


SOLVERBENCH_COLUMNS = [
    "sweep", "arm", "budget_secs", "profile", "pod_count", "md_count", "slots_level",
    "rep", "instance_id", "num_pods", "num_mds", "total_slots", "num_workers",
    "success", "status", "objective", "solve_time_secs", "wall_time_secs",
    "unplaced_pods", "beam_used", "beam_fallback_used", "attempts_run",
    "winning_attempt", "nodes_provisioned", "cost_per_hour", "best_objective_bound",
    "optimality_gap",
]


def load_solver_bench(kind, results_dir=RESULTS_DIR) -> pd.DataFrame:
    """Offline solver benchmark CSVs written by benchmarks/solver-bench
    (make solver-bench-rq2 / solver-bench-rq3). kind is "rq2" or "rq3".
    These have their own glob: BATCH_NAME_RE only matches harness kinds.
    A sweep is deterministic and complete, so only the newest file counts;
    concatenating reruns would duplicate (instance, arm, rep) keys."""
    files = sorted(results_dir.glob(f"solverbench_*_{kind}.csv"))
    if not files:
        return pd.DataFrame(columns=SOLVERBENCH_COLUMNS)
    df = pd.read_csv(files[-1])
    for col in ("success", "beam_used", "beam_fallback_used"):
        df[col] = df[col].astype(str).str.lower() == "true"
    return df.reset_index(drop=True)
