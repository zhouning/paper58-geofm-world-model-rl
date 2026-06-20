from pathlib import Path

from scripts.paper58_benchmark.make_batch23_urban_contrast import (
    BATCH2_URBAN_AREAS,
    BATCH3_URBAN_AREAS,
    build_urban_contrast_diagnostics,
)


class FakeDiagnostics:
    def __init__(self):
        self.calls = []

    def make_batch2_alignment_table(self, **kwargs):
        self.calls.append(("alignment", kwargs["areas"]))
        return [{"area": area, "raw_change_f1": 0.1, "best_shift_change_f1": 0.2} for area in kwargs["areas"]]

    def make_embedding_decoder_audit_table(self, **kwargs):
        self.calls.append(("decoder_audit", kwargs["areas"]))
        return [{"area": area, "end_decode_accuracy": 0.5} for area in kwargs["areas"]]

    def make_decoder_true_end_confidence_table(self, **kwargs):
        self.calls.append(("true_end_confidence", kwargs["areas"]))
        return [{"area": area, "true_end_class": 11, "mean_true_end_prob": 0.1} for area in kwargs["areas"]]

    def make_forecast_true_end_confidence_table(self, **kwargs):
        self.calls.append(("forecast_confidence", kwargs["areas"]))
        return [{"area": area, "true_end_class": 11, "forecast_mean_true_end_prob": 0.2} for area in kwargs["areas"]]

    def make_transition_table(self, **kwargs):
        self.calls.append(("transition_table", kwargs["area"]))
        return [{"area": kwargs["area"], "source": "reference_change", "start_class": 5, "end_class": 11}]

    def make_transition_fate_table(self, **kwargs):
        self.calls.append(("transition_fate", kwargs["area"]))
        return [{"area": kwargs["area"], "true_transition": "5->11", "n_true_pixels": 3}]

    def make_forecast_transition_fate_table(self, **kwargs):
        self.calls.append(("forecast_transition_fate", kwargs["area"]))
        return [{"area": kwargs["area"], "true_transition": "5->11", "forecast_mean_true_end_prob": 0.2}]


def test_build_urban_contrast_diagnostics_runs_all_urban_area_audits(tmp_path: Path):
    fake = FakeDiagnostics()

    summary = build_urban_contrast_diagnostics(
        output_dir=tmp_path,
        decoder=object(),
        model=object(),
        diagnostics=fake,
    )

    expected_areas = BATCH2_URBAN_AREAS + BATCH3_URBAN_AREAS
    assert summary == {
        "n_urban_areas": 6,
        "batch2_urban_areas": BATCH2_URBAN_AREAS,
        "batch3_urban_areas": BATCH3_URBAN_AREAS,
        "output_dir": str(tmp_path),
    }
    assert ("alignment", expected_areas) in fake.calls
    assert ("decoder_audit", expected_areas) in fake.calls
    assert ("true_end_confidence", expected_areas) in fake.calls
    assert ("forecast_confidence", expected_areas) in fake.calls
    assert ("transition_table", "xiong_an_fringe_holdout") in fake.calls
    assert ("transition_fate", "fuzhou_delta_urban_holdout") in fake.calls
    assert (tmp_path / "urban_contrast_summary.txt").exists()
    assert (tmp_path / "urban_transition_fate_all.csv").exists()
