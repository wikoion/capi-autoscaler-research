import re
import sys
from pathlib import Path

import pandas as pd

from capi_bench_analysis.loading import (
    load_summaries, load_solver_metrics, parse_instance_counts,
)

HEADER = ("autoscaler,scenario,pod_count,profile,run_num,run_id,latency_s,"
          "first_unschedulable_s,all_scheduled_s,all_running_s,worker_nodes,"
          "workload_cpu_util_pct,workload_mem_util_pct,cost_per_hour,"
          "instance_counts,timed_out")


def row(autoscaler, scenario, run_num, latency="100.0", timed_out="False",
        suffix="abc123"):
    profile, pods = scenario.rsplit("-", 1)
    return (f"{autoscaler},{scenario},{pods},{profile},{run_num},"
            f"{scenario}-{run_num}-{suffix},{latency},5.0,50.0,{latency},10,"
            f"70.0,60.0,1.5,m5.large=10,{timed_out}")


def test_runs_pool_across_files(tmp_path):
    f1 = tmp_path / "benchmark_20260728_100000_summary.csv"
    f1.write_text("\n".join([
        HEADER,
        row("capi-autoscaler", "prod-mix-100", 1, latency="200.0"),
        row("capi-autoscaler", "prod-mix-100", 2, latency="210.0"),
        row("cas", "prod-mix-100", 1, latency="300.0"),
    ]) + "\n")
    f2 = tmp_path / "benchmark_20260728_110000_summary.csv"
    f2.write_text("\n".join([
        HEADER,
        row("capi-autoscaler", "prod-mix-100", 1, latency="150.0", suffix="def456"),
        row("capi-autoscaler", "prod-mix-100", 2, latency="160.0", suffix="def456"),
    ]) + "\n")

    df = load_summaries(results_dir=tmp_path)

    # No supersession: all four capi runs count towards the mean
    capi = df[df["autoscaler"] == "capi-autoscaler"]
    assert len(capi) == 4
    assert sorted(capi["latency_s"]) == [150.0, 160.0, 200.0, 210.0]
    assert len(df[df["autoscaler"] == "cas"]) == 1


def test_cohort_labels_by_file_date(tmp_path):
    f1 = tmp_path / "benchmark_20260830_100000_summary.csv"
    f1.write_text("\n".join([
        HEADER,
        row("capi-autoscaler", "prod-mix-100", 1),
        row("cas", "prod-mix-100", 1),
    ]) + "\n")
    f2 = tmp_path / "benchmark_20260831_100000_summary.csv"
    f2.write_text("\n".join([
        HEADER,
        row("capi-autoscaler", "prod-mix-100", 1, latency="120.0", suffix="def456"),
    ]) + "\n")

    df = load_summaries(results_dir=tmp_path)

    capi = df[df["autoscaler"] == "capi-autoscaler"]
    assert sorted(capi["cohort"]) == ["placementcontrol", "pre-placementcontrol"]
    assert df.loc[df["autoscaler"] == "cas", "cohort"].tolist() == ["baseline"]


def test_all_prefixes_load(tmp_path):
    f1 = tmp_path / "placement9_20260728_100000_summary.csv"
    f1.write_text("\n".join([HEADER, row("cas", "prod-mix-100", 1, latency="300.0")]) + "\n")
    f2 = tmp_path / "benchmark_20260728_110000_summary.csv"
    f2.write_text("\n".join([HEADER, row("cas", "prod-mix-100", 2, latency="150.0")]) + "\n")

    df = load_summaries(results_dir=tmp_path)

    assert len(df) == 2
    assert sorted(df["latency_s"]) == [150.0, 300.0]


def test_duplicate_prefix_copies_do_not_double_rows(tmp_path):
    content = "\n".join([
        HEADER,
        row("capi-autoscaler", "prod-mix-100", 1),
        row("capi-autoscaler", "prod-mix-100", 2),
    ]) + "\n"
    (tmp_path / "benchmark_20260831_100000_summary.csv").write_text(content)
    (tmp_path / "placement8_20260831_100000_summary.csv").write_text(content)

    df = load_summaries(results_dir=tmp_path)

    assert len(df) == 2


def test_na_and_timed_out_parsing(tmp_path):
    f = tmp_path / "benchmark_20260728_100000_summary.csv"
    f.write_text("\n".join([
        HEADER,
        row("cas", "prod-mix-50", 1, latency="N/A", timed_out="True"),
    ]) + "\n")
    df = load_summaries(results_dir=tmp_path)
    assert pd.isna(df.iloc[0]["latency_s"])
    assert bool(df.iloc[0]["timed_out"]) is True
    assert df.iloc[0]["pod_count"] == 50


def test_parse_instance_counts():
    assert parse_instance_counts("c5.2xlarge=1;m5.4xlarge=8") == {
        "c5.2xlarge": 1, "m5.4xlarge": 8,
    }
    assert parse_instance_counts("") == {}


def test_solver_metrics_empty_when_missing(tmp_path):
    df = load_solver_metrics(results_dir=tmp_path)
    assert df.empty
    assert "solve_time_secs" in df.columns


def test_solver_result_regex_matches_rust_log_line():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "capi-autoscaler" / "benchmarks" / "harness"))
    from benchmark.solver_logs import SOLVER_RESULT_RE
    line = ("2026-08-29T10:00:00Z  INFO capi_autoscaler::optimiser: Solver result: "
            "status=Optimal objective=-1234.5 unplaced=0 used_beam=false solve_time=2.41s")
    m = SOLVER_RESULT_RE.search(line)
    assert m is not None
    assert m.group("status") == "Optimal"
    assert float(m.group("objective")) == -1234.5
    assert m.group("used_beam") == "false"
    assert float(m.group("solve_time")) == 2.41
