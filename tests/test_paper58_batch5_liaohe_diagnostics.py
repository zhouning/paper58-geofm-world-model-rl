import csv
from pathlib import Path

from scripts.paper58_benchmark.make_batch5_liaohe_diagnostics import (
    FOCUS_AREA,
    SECONDARY_RISK_AREAS,
    WETLAND_COMPARISON_AREAS,
    build_liaohe_diagnostics,
)


def _write_batch5_metrics(path: Path) -> None:
    path.mkdir(parents=True)
    fieldnames = [
        "area",
        "start_year",
        "end_year",
        "tier",
        "stratum",
        "true_change_pixels",
        "model_change_f1",
        "primary_change_advantage",
        "spatial_shuffle_change_f1",
        "spatial_change_advantage",
    ]
    rows = [
        {
            "area": FOCUS_AREA,
            "start_year": 2020,
            "end_year": 2021,
            "tier": "tier1",
            "stratum": "Wetland",
            "true_change_pixels": 62,
            "model_change_f1": 0.179,
            "primary_change_advantage": -0.009,
            "spatial_shuffle_change_f1": 0.216,
            "spatial_change_advantage": -0.037,
        },
        {
            "area": SECONDARY_RISK_AREAS[0],
            "start_year": 2020,
            "end_year": 2021,
            "tier": "tier1",
            "stratum": "Urban",
            "true_change_pixels": 10,
            "model_change_f1": 0.054,
            "primary_change_advantage": -0.033,
            "spatial_shuffle_change_f1": 0.044,
            "spatial_change_advantage": 0.010,
        },
    ]
    with (path / "benchmark_metrics_by_pair.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FakeDiagnostics:
    def __init__(self):
        self.calls = []

    def make_batch2_alignment_table(self, **kwargs):
        self.calls.append(("alignment", kwargs["areas"], kwargs["output_filename"]))
        return [
            {
                "area": area,
                "raw_change_f1": 0.10,
                "best_shift_change_f1": 0.20,
                "best_dy": 1,
                "best_dx": -1,
            }
            for area in kwargs["areas"]
        ]

    def make_embedding_decoder_audit_table(self, **kwargs):
        self.calls.append(("decoder_audit", kwargs["areas"], kwargs["output_filename"]))
        return [{"area": area, "end_decode_accuracy": 0.50} for area in kwargs["areas"]]

    def make_decoder_true_end_confidence_table(self, **kwargs):
        self.calls.append(("true_end_confidence", kwargs["areas"], kwargs["output_filename"]))
        return [{"area": area, "true_end_class": 11, "mean_true_end_prob": 0.10} for area in kwargs["areas"]]

    def make_forecast_true_end_confidence_table(self, **kwargs):
        self.calls.append(("forecast_confidence", kwargs["areas"], kwargs["output_filename"]))
        return [
            {
                "area": area,
                "true_end_class": 11,
                "forecast_mean_true_end_prob": 0.08,
                "mean_true_end_prob_delta": -0.02,
            }
            for area in kwargs["areas"]
        ]

    def make_transition_table(self, **kwargs):
        self.calls.append(("transition", kwargs["area"]))
        return [{"source": "reference_change", "start_class": 5, "end_class": 11, "n_pixels": 4}]

    def make_transition_fate_table(self, **kwargs):
        self.calls.append(("transition_fate", kwargs["area"]))
        return [
            {
                "true_transition": "5->11",
                "n_true_pixels": 4,
                "decoded_end_top": "5:4",
                "model_end_top": "5:4",
                "mean_true_end_prob": 0.10,
            }
        ]

    def make_shifted_transition_fate_table(self, **kwargs):
        self.calls.append(("shifted_transition_fate", kwargs["area"], kwargs["shift_dy"], kwargs["shift_dx"]))
        return [
            {
                "true_transition": "5->11",
                "n_true_pixels": 4,
                "raw_model_end_top": "5:4",
                "shifted_model_end_top": "11:2;5:2",
                "raw_match_pixels": 0,
                "shifted_match_pixels": 2,
            }
        ]

    def make_forecast_transition_fate_table(self, **kwargs):
        self.calls.append(("forecast_transition_fate", kwargs["area"]))
        return [
            {
                "true_transition": "5->11",
                "n_true_pixels": 4,
                "observed_end_top": "5:4",
                "forecast_end_top": "5:4",
                "observed_mean_true_end_prob": 0.10,
                "forecast_mean_true_end_prob": 0.08,
                "mean_true_end_prob_delta": -0.02,
            }
        ]


def test_build_liaohe_diagnostics_runs_focus_comparators_and_secondary_risk(tmp_path: Path):
    results_dir = tmp_path / "benchmark_results_batch5"
    output_dir = tmp_path / "diagnostics"
    _write_batch5_metrics(results_dir)
    fake = FakeDiagnostics()

    summary = build_liaohe_diagnostics(
        batch5_results_dir=results_dir,
        output_dir=output_dir,
        decoder=object(),
        model=object(),
        diagnostics=fake,
    )

    diagnostic_areas = [FOCUS_AREA, *WETLAND_COMPARISON_AREAS, *SECONDARY_RISK_AREAS]
    assert summary["focus_area"] == FOCUS_AREA
    assert summary["focus_metrics"]["primary_change_advantage"] == -0.009
    assert summary["secondary_risks"][0]["area"] == SECONDARY_RISK_AREAS[0]
    assert ("alignment", diagnostic_areas, "batch5_liaohe_spatial_alignment_shift.csv") in fake.calls
    assert ("decoder_audit", diagnostic_areas, "batch5_liaohe_embedding_decoder_audit.csv") in fake.calls
    assert ("transition_fate", FOCUS_AREA) in fake.calls
    assert ("shifted_transition_fate", FOCUS_AREA, 1, -1) in fake.calls
    assert (output_dir / "batch5_liaohe_transition_counts_all.csv").exists()
    assert (output_dir / "batch5_liaohe_transition_fate_all.csv").exists()
    assert (output_dir / "batch5_liaohe_summary.json").exists()
    assert (output_dir / "batch5_liaohe_summary.txt").exists()
