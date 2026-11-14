"""Auto-written per-scenario prose. Comparative claims are suppressed when
either side has n < 2 or the confidence intervals overlap."""

import math

import pandas as pd

COMPARE_METRICS = [
    ("latency_s", "end-to-end latency", "s", True),
    ("workload_cpu_util_pct", "CPU utilisation", "%", False),
    ("workload_mem_util_pct", "memory utilisation", "%", False),
    ("cost_per_hour", "fleet cost", " $/hour", True),
]


def _fmt(mean, ci, n, unit):
    if math.isnan(mean):
        return "no data"
    ci_txt = f" ± {ci:.1f}" if not math.isnan(ci) else ""
    return f"{mean:.1f}{ci_txt}{unit} (n={n})"


def _cis_overlap(m1, c1, m2, c2):
    if math.isnan(c1) or math.isnan(c2):
        return True  # can't claim a difference without both CIs
    return (m1 - c1) <= (m2 + c2) and (m2 - c2) <= (m1 + c1)


def summarise_scenario(score_df: pd.DataFrame, scenario_row_filter: dict) -> str:
    """One markdown block for a scenario. score_df is a score_table over
    (autoscaler, pod_count)."""
    sub = score_df
    for col, val in scenario_row_filter.items():
        sub = sub[sub[col] == val]
    if sub.empty:
        return ""
    lines = []
    capi = sub[sub["autoscaler"] == "capi-autoscaler"]
    for _, other in sub[sub["autoscaler"] != "capi-autoscaler"].iterrows():
        for metric, label, unit, lower_better in COMPARE_METRICS:
            if capi.empty:
                continue
            c = capi.iloc[0]
            cm, cc, cn = c[f"{metric}_mean"], c[f"{metric}_ci95"], c["n_ok"]
            om, oc, on = other[f"{metric}_mean"], other[f"{metric}_ci95"], other["n_ok"]
            if math.isnan(cm) or math.isnan(om):
                continue
            base = (f"- {label}: capi-autoscaler {_fmt(cm, cc, cn, unit)} vs "
                    f"{other['autoscaler']} {_fmt(om, oc, on, unit)}")
            if cn < 2 or on < 2 or _cis_overlap(cm, cc, om, oc):
                lines.append(base + "; difference within CI overlap, no claim made.")
            else:
                better = (cm < om) == lower_better
                pct = abs(cm - om) / om * 100
                verdict = "better" if better else "worse"
                lines.append(base + f"; capi-autoscaler {pct:.1f}% {verdict}.")
    timeouts = sub[sub["n_timed_out"] > 0]
    for _, r in timeouts.iterrows():
        lines.append(f"- {r['autoscaler']} timed out in {r['n_timed_out']}/{r['n_total']} runs.")
    return "\n".join(lines)


def no_data_note(section: str, reason: str) -> str:
    return f"**No data yet for {section}.** {reason}"
