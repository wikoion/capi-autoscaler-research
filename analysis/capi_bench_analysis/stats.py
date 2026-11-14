"""Mean/CI aggregation matching the EPP methodology (means with 95% CIs)."""

import math

import numpy as np
import pandas as pd
from scipy import stats as sps


def mean_ci(values, confidence=0.95):
    """(mean, ci_half_width, n) over non-NaN values; half width is NaN for n < 2."""
    vals = pd.Series(values).dropna().astype(float)
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan, 0)
    mean = vals.mean()
    if n < 2:
        return (mean, math.nan, n)
    half = sps.t.ppf((1 + confidence) / 2, n - 1) * vals.std(ddof=1) / math.sqrt(n)
    return (mean, half, n)


def score_table(df: pd.DataFrame, group_cols, metrics, confidence=0.95) -> pd.DataFrame:
    """Per-group means and CIs. Timed-out runs are excluded from the means but
    counted, matching the ok filter in benchmark.cli summarise."""
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ok = group[~group["timed_out"]]
        row = dict(zip(group_cols, keys))
        row["n_ok"] = len(ok)
        row["n_total"] = len(group)
        row["n_timed_out"] = int(group["timed_out"].sum())
        for m in metrics:
            mean, half, n = mean_ci(ok[m], confidence)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def relative_gap(df: pd.DataFrame, instance_col="instance_id",
                 objective_col="objective") -> pd.Series:
    """Percent gap to the best fully-placed objective for the same instance.
    NaN for rows with unplaced pods (the unplaced penalty is fleet-cost scale
    and would swamp the gap) and for instances with no fully-placed solve."""
    placed = df["success"] & (df["unplaced_pods"] == 0)
    best = df[placed].groupby(instance_col)[objective_col].min()
    ref = df[instance_col].map(best)
    gap = (df[objective_col] - ref) / ref * 100
    return gap.where(placed)


def provisioning_efficiency(ts_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Requested/allocatable cpu and mem at each run's steady point (first
    all-running sample; see loading.steady_points)."""
    from .loading import steady_points

    if ts_df.empty:
        return pd.DataFrame(columns=["autoscaler", "scenario", "run_id",
                                     "cpu_efficiency", "mem_efficiency"])
    sp = steady_points(ts_df, summary_df)
    out = sp[["autoscaler", "scenario", "run_id"]].copy()
    out["cpu_efficiency"] = sp["workload_cpu_requested_m"] / sp["worker_cpu_allocatable_m"].replace(0, np.nan)
    out["mem_efficiency"] = sp["workload_mem_requested_mi"] / sp["worker_mem_allocatable_mi"].replace(0, np.nan)
    return out.reset_index(drop=True)
