from edge_bench.utils.metrics import latency_stats


def test_latency_stats_reports_expected_percentiles() -> None:
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert round(stats["mean_ms"], 3) == 3.0
    assert round(stats["p50_ms"], 3) == 3.0
    assert round(stats["p95_ms"], 3) == 4.8
