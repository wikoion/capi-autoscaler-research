"""Matplotlib figures for the report. Colour is assigned per autoscaler in a
fixed order (never cycled), so an autoscaler keeps its hue across every figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical assignment, light-mode palette (report is printed on white).
# The pre/post pair only appears together in the RQ4 pre-vs-post charts; post
# keeps the capi blue, pre is a lighter step of the same hue (validated pair).
AUTOSCALER_COLOURS = {
    "capi-autoscaler": "#2a78d6",
    "cas": "#1baf7a",
    "karpenter": "#eda100",
    "karpenter-consolidation": "#b87700",
    "capi (pre-placementcontrol)": "#6ba3ea",
    "capi (placementcontrol)": "#2a78d6",
}
AUTOSCALER_ORDER = list(AUTOSCALER_COLOURS)

# Display-only renames; data and colours stay keyed on the harness names
AUTOSCALER_LABELS = {"cas": "ca", "karpenter-consolidation": "karpenter (consolidation)"}


def display_name(a):
    return AUTOSCALER_LABELS.get(a, a)

TEXT_PRIMARY = "#333333"
GRID_COLOUR = "#dddddd"

METRIC_LABELS = {
    "latency_s": "End-to-end latency (s)",
    "first_unschedulable_s": "First unschedulable (s)",
    "all_scheduled_s": "All pods scheduled (s)",
    "all_running_s": "All pods running (s)",
    "worker_nodes": "Worker nodes",
    "workload_cpu_util_pct": "CPU utilisation (%)",
    "workload_mem_util_pct": "Memory utilisation (%)",
    "cost_per_hour": "Fleet cost ($/hour)",
    "pending_pods": "Pending pods",
    "solve_time_secs": "Solve time (s)",
}


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_PRIMARY)


def _present(df, col="autoscaler"):
    return [a for a in AUTOSCALER_ORDER if a in set(df[col])]


def _comparison_bars_ax(ax, table: pd.DataFrame, metric: str):
    autoscalers = _present(table)
    pod_counts = sorted(table["pod_count"].dropna().unique())
    width = 0.8 / max(len(autoscalers), 1)
    x = np.arange(len(pod_counts))

    for i, a in enumerate(autoscalers):
        sub = table[table["autoscaler"] == a].set_index("pod_count")
        means = [sub[f"{metric}_mean"].get(p, np.nan) for p in pod_counts]
        errs = [sub[f"{metric}_ci95"].get(p, np.nan) for p in pod_counts]
        pos = x + (i - (len(autoscalers) - 1) / 2) * width
        clean_errs = [0 if np.isnan(e) else e for e in errs]
        bars = ax.bar(pos, means, width * 0.9, label=display_name(a), color=AUTOSCALER_COLOURS[a],
                      yerr=clean_errs, capsize=3, ecolor=TEXT_PRIMARY, zorder=2)
        for rect, mean, err, p in zip(bars, means, clean_errs, pod_counts):
            n_ok = sub["n_ok"].get(p, 0)
            n_total = sub["n_total"].get(p, 0)
            if not np.isnan(mean):
                # Label above the error bar cap, not the bar top; single-run
                # means have no CI, so say so
                label = f"{mean:.1f}" + ("\nn=1" if n_ok == 1 else "")
                ax.annotate(label, (rect.get_x() + rect.get_width() / 2, mean + err),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=TEXT_PRIMARY)
            elif n_total > 0:
                # Runs exist but none completed; mark the empty slot
                ax.annotate(f"timed out {int(n_ok)}/{int(n_total)}",
                            (rect.get_x() + rect.get_width() / 2, 0),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", rotation=90,
                            fontsize=7, color=TEXT_PRIMARY)

    ax.set_xticks(x, [str(p) for p in pod_counts])
    # These metrics can't be negative; wide CIs must not drag the axis below 0
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pod count", color=TEXT_PRIMARY)
    _style_axes(ax)


def comparison_bars(table: pd.DataFrame, metric: str, title: str):
    """Grouped bars per pod_count, one bar per autoscaler, 95% CI error bars,
    direct value labels on every bar."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _comparison_bars_ax(ax, table, metric)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    if len(_present(table)) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def comparison_bars_composite(tables: dict, metric: str, title: str):
    """One panel per profile from comparison_bars, shared y axis, one legend.
    tables maps profile name to its score table."""
    profiles = [p for p in PROFILE_COLOURS if p in tables] or list(tables)
    fig, axes = plt.subplots(1, len(profiles), figsize=(13, 4), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, profile in zip(axes, profiles):
        _comparison_bars_ax(ax, tables[profile], metric)
        ax.set_title(profile, color=TEXT_PRIMARY, fontsize=10)
    # Shared y locks in the first panel's top; set it from the tallest
    # mean+CI across all panels so no bar or value label is clipped
    top = max((t[f"{metric}_mean"] + t[f"{metric}_ci95"].fillna(0)).max()
              for t in tables.values())
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric), color=TEXT_PRIMARY)
    # Bars fill every panel, so the legend goes above the axes, not inside.
    # Four or more entries wrap to two columns so they clear the title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8,
               ncol=len(labels) if len(labels) <= 3 else 2,
               loc="outside upper right")
    fig.suptitle(title, color=TEXT_PRIMARY)
    return fig


def prepost_composite(tables: dict, metrics: list, pod_count, title: str):
    """RQ4 in one figure: rows are metrics, columns are profiles, each panel
    two bars (pre vs post placementcontrol) at a single pod count. tables maps
    profile name to a prepost score table."""
    profiles = [p for p in PROFILE_COLOURS if p in tables] or list(tables)
    order = ["capi (pre-placementcontrol)", "capi (placementcontrol)"]
    fig, axes = plt.subplots(len(metrics), len(profiles), figsize=(13, 7),
                             sharey="row", layout="constrained")
    axes = np.atleast_2d(axes)
    for i, metric in enumerate(metrics):
        row_top = 0.0
        for j, profile in enumerate(profiles):
            ax = axes[i][j]
            sub = tables[profile]
            sub = sub[sub["pod_count"] == pod_count].set_index("autoscaler")
            for k, a in enumerate(order):
                mean = sub[f"{metric}_mean"].get(a, np.nan)
                err = sub[f"{metric}_ci95"].get(a, np.nan)
                err = 0 if np.isnan(err) else err
                ax.bar(k, mean, 0.6, color=AUTOSCALER_COLOURS[a], yerr=err,
                       capsize=3, ecolor=TEXT_PRIMARY, zorder=2)
                if not np.isnan(mean):
                    n_ok = sub["n_ok"].get(a, 0)
                    label = f"{mean:.1f}" + ("\nn=1" if n_ok == 1 else "")
                    ax.annotate(label, (k, mean + err), xytext=(0, 3),
                                textcoords="offset points", ha="center",
                                fontsize=8, color=TEXT_PRIMARY)
                    row_top = max(row_top, mean + err)
            ax.set_xticks([0, 1], ["pre", "post"])
            if i == 0:
                ax.set_title(profile, color=TEXT_PRIMARY, fontsize=10)
            _style_axes(ax)
        axes[i][0].set_ylim(0, row_top * 1.15)
        axes[i][0].set_ylabel(METRIC_LABELS.get(metric, metric), color=TEXT_PRIMARY)
    fig.suptitle(title, color=TEXT_PRIMARY)
    return fig


def add_footnote(fig, text: str):
    """Small grey caption under the axes; bbox_inches='tight' keeps it in the
    saved figure."""
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=7.5, color="#666666")
    return fig


def timeseries_overlay(ts_df: pd.DataFrame, metric: str, scenario: str, title: str):
    """Per autoscaler: mean line over runs on a common elapsed grid, with a
    shaded min-max band."""
    sub = ts_df[ts_df["scenario"] == scenario]
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in _present(sub):
        runs = sub[sub["autoscaler"] == a]
        grid = np.arange(0, runs["elapsed_s"].max() + 1, 5)
        interpolated = [
            np.interp(grid, run.sort_values("elapsed_s")["elapsed_s"], run.sort_values("elapsed_s")[metric])
            for _, run in runs.groupby("run_id")
        ]
        stacked = np.vstack(interpolated)
        ax.plot(grid, stacked.mean(axis=0), color=AUTOSCALER_COLOURS[a], linewidth=2,
                label=display_name(a))
        ax.fill_between(grid, stacked.min(axis=0), stacked.max(axis=0),
                        color=AUTOSCALER_COLOURS[a], alpha=0.15, linewidth=0)
    ax.set_xlabel("Elapsed (s)", color=TEXT_PRIMARY)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    if sub["autoscaler"].nunique() > 1:
        ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def requested_vs_allocatable(ts_df: pd.DataFrame, scenario: str, resource: str, title: str):
    """One chart per resource ('cpu' or 'mem'): solid allocatable line and
    dashed requested line per autoscaler, mean over runs."""
    req_col = f"workload_{resource}_requested_m" if resource == "cpu" else "workload_mem_requested_mi"
    alloc_col = f"worker_{resource}_allocatable_m" if resource == "cpu" else "worker_mem_allocatable_mi"
    unit = "mCPU" if resource == "cpu" else "MiB"
    sub = ts_df[ts_df["scenario"] == scenario]
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in _present(sub):
        runs = sub[sub["autoscaler"] == a]
        grid = np.arange(0, runs["elapsed_s"].max() + 1, 5)
        for col, style, label in ((alloc_col, "-", "allocatable"), (req_col, "--", "requested")):
            stacked = np.vstack([
                np.interp(grid, run.sort_values("elapsed_s")["elapsed_s"], run.sort_values("elapsed_s")[col])
                for _, run in runs.groupby("run_id")
            ])
            ax.plot(grid, stacked.mean(axis=0), style, color=AUTOSCALER_COLOURS[a],
                    linewidth=2, label=f"{display_name(a)} {label}")
    ax.set_xlabel("Elapsed (s)", color=TEXT_PRIMARY)
    ax.set_ylabel(f"{resource.upper()} ({unit})", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def instance_mix_stacked(long_df: pd.DataFrame, scenario: str, title: str, by_cost=False):
    """Stacked bars of the mean instance mix per autoscaler for one scenario.
    Instance types use a single-hue sequential ramp (magnitude of one fleet,
    not identities across figures); a legend names each segment."""
    sub = long_df[long_df["scenario"] == scenario]
    value = "hourly_cost" if by_cost else "count"
    pivot = (sub.groupby(["autoscaler", "instance_type"])[value].mean().unstack(fill_value=0))
    pivot = pivot.reindex([a for a in AUTOSCALER_ORDER if a in pivot.index])
    pivot.index = [display_name(a) for a in pivot.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    types = sorted(pivot.columns)
    shades = plt.cm.Blues(np.linspace(0.35, 0.9, max(len(types), 2)))
    bottom = np.zeros(len(pivot))
    for shade, t in zip(shades, types):
        vals = pivot[t].to_numpy()
        ax.bar(pivot.index, vals, 0.6, bottom=bottom, label=t, color=shade,
               edgecolor="white", linewidth=1, zorder=2)
        bottom += vals
    ax.set_ylabel("Mean fleet $/hour" if by_cost else "Mean node count", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def solver_scaling(solver_df: pd.DataFrame, summary_df: pd.DataFrame, title: str):
    """Solve time vs pod count for capi-autoscaler runs; beam-search fallback
    solves marked with an open marker."""
    merged = solver_df.merge(summary_df[["run_id", "pod_count"]], on="run_id", how="left")
    fig, ax = plt.subplots(figsize=(7, 4))
    for used_beam, marker, label in ((False, "o", "CP-SAT"), (True, "^", "beam fallback")):
        pts = merged[merged["used_beam"] == used_beam]
        if not pts.empty:
            ax.scatter(pts["pod_count"], pts["solve_time_secs"], s=60, marker=marker,
                       facecolors=AUTOSCALER_COLOURS["capi-autoscaler"] if not used_beam else "none",
                       edgecolors=AUTOSCALER_COLOURS["capi-autoscaler"], label=label, zorder=2)
    ax.set_xlabel("Pod count", color=TEXT_PRIMARY)
    ax.set_ylabel(METRIC_LABELS["solve_time_secs"], color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig


PROFILE_COLOURS = {
    "unconstrained": "#2a78d6",
    "light-constrained": "#1baf7a",
    "heavy-constrained": "#eda100",
}


def _offline_scaling_ax(ax, df: pd.DataFrame):
    md_counts = sorted(df["md_count"].unique())
    shades = plt.cm.Blues(np.linspace(0.45, 0.9, max(len(md_counts), 2)))
    for shade, mdc in zip(shades, md_counts):
        g = df[df["md_count"] == mdc].groupby("pod_count")["wall_time_secs"]
        ax.plot(g.mean().index, g.mean(), color=shade, linewidth=2, marker="o",
                label=f"{mdc} MDs", zorder=2)
        ax.fill_between(g.mean().index, g.min(), g.max(), color=shade, alpha=0.15, linewidth=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pod count", color=TEXT_PRIMARY)
    _style_axes(ax)


def offline_scaling(df: pd.DataFrame, title: str):
    """Offline RQ2: mean solve wall time vs pod count, one line per MD count
    with a min-max band. Expects one profile's rows at a single slots level."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _offline_scaling_ax(ax, df)
    ax.set_ylabel("Solve wall time (s)", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def offline_scaling_composite(panels: dict, title: str):
    """One offline_scaling panel per profile, shared y axis, one legend.
    panels maps profile name to its rows at a single slots level."""
    profiles = [p for p in PROFILE_COLOURS if p in panels] or list(panels)
    fig, axes = plt.subplots(1, len(profiles), figsize=(13, 4), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, profile in zip(axes, profiles):
        _offline_scaling_ax(ax, panels[profile])
        ax.set_title(profile, color=TEXT_PRIMARY, fontsize=10)
    axes[0].set_ylabel("Solve wall time (s)", color=TEXT_PRIMARY)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=TEXT_PRIMARY)
    return fig


def _offline_feasibility_ax(ax, df: pd.DataFrame):
    levels = [l for l in ("low", "med", "high") if l in set(df["slots_level"])]
    pod_counts = sorted(df["pod_count"].unique())
    width = 0.8 / max(len(pod_counts), 1)
    x = np.arange(len(levels))
    shades = plt.cm.Blues(np.linspace(0.45, 0.9, max(len(pod_counts), 2)))
    for i, (shade, p) in enumerate(zip(shades, pod_counts)):
        means = [df[(df["slots_level"] == l) & (df["pod_count"] == p)]["unplaced_pods"].mean()
                 for l in levels]
        pos = x + (i - (len(pod_counts) - 1) / 2) * width
        ax.bar(pos, means, width * 0.9, label=f"{p} pods", color=shade, zorder=2)
    ax.set_xticks(x, levels)
    ax.set_xlabel("Candidate slots level", color=TEXT_PRIMARY)
    _style_axes(ax)


def offline_feasibility(df: pd.DataFrame, title: str):
    """Offline RQ2: mean unplaced pods per slots level, grouped by pod count.
    Expects one profile's rows at a single MD count."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _offline_feasibility_ax(ax, df)
    ax.set_ylabel("Mean unplaced pods", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def offline_feasibility_composite(panels: dict, title: str):
    """One offline_feasibility panel per profile, shared y axis, one legend.
    panels maps profile name to its rows at a single MD count."""
    profiles = [p for p in PROFILE_COLOURS if p in panels] or list(panels)
    fig, axes = plt.subplots(1, len(profiles), figsize=(13, 4), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, profile in zip(axes, profiles):
        _offline_feasibility_ax(ax, panels[profile])
        ax.set_title(profile, color=TEXT_PRIMARY, fontsize=10)
    axes[0].set_ylabel("Mean unplaced pods", color=TEXT_PRIMARY)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=TEXT_PRIMARY)
    return fig


def offline_gap(df: pd.DataFrame, title: str):
    """Offline RQ2: mean proven optimality gap vs pod count per profile, with a
    min-max band. Expects gap_pct set only for fully placed CP-SAT solves;
    0 means the solve proved optimality inside the budget."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for profile, colour in PROFILE_COLOURS.items():
        g = df[df["profile"] == profile].groupby("pod_count")["gap_pct"]
        if g.mean().dropna().empty:
            continue
        ax.plot(g.mean().index, g.mean(), color=colour, linewidth=2, marker="o",
                label=profile, zorder=2)
        ax.fill_between(g.mean().index, g.min(), g.max(), color=colour, alpha=0.15, linewidth=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pod count", color=TEXT_PRIMARY)
    ax.set_ylabel("Proven optimality gap (%)", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def budget_quality(df: pd.DataFrame, title: str):
    """Offline RQ3: mean quality gap vs CP-SAT budget per profile, log-x.
    Beam-only drawn as a dashed horizontal reference where its gap is defined
    (fully placed); budgets where no solve was fully placed have no point."""
    cpsat = df[df["arm"].str.startswith("cpsat-")]
    beam = df[df["arm"] == "beam"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for profile, colour in PROFILE_COLOURS.items():
        g = cpsat[cpsat["profile"] == profile].groupby("budget_secs")["gap_pct"].mean().dropna()
        if not g.empty:
            ax.plot(g.index, g.values, color=colour, linewidth=2, marker="o",
                    label=profile, zorder=2)
        beam_gap = beam[beam["profile"] == profile]["gap_pct"].mean()
        if not np.isnan(beam_gap):
            ax.axhline(beam_gap, color=colour, linestyle="--", alpha=0.6, linewidth=1.5)
    ax.set_xscale("log")
    budgets = sorted(cpsat["budget_secs"].unique())
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("CP-SAT budget (s)", color=TEXT_PRIMARY)
    ax.set_ylabel("Gap to best solution (%)", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig
