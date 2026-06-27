import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from scripts.paper58_benchmark.make_las_visual_report import make_las_visual_report


def test_make_las_visual_report_writes_metric_and_spatial_figures(tmp_path: Path):
    start = np.array([[5, 7], [5, 7]], dtype=np.int32)
    end = np.array([[5, 7], [7, 5]], dtype=np.int32)
    direct = np.array([[7, 5], [7, 5]], dtype=np.int32)
    flus = np.array([[5, 7], [5, 7]], dtype=np.int32)
    las = np.array([[5, 7], [7, 5]], dtype=np.int32)

    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    direct_path = tmp_path / "direct.npy"
    flus_path = tmp_path / "flus.npy"
    np.save(start_path, start)
    np.save(end_path, end)
    np.save(direct_path, direct)
    np.save(flus_path, flus)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "toy_area",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        "bbox": [120.0, 30.0, 120.1, 30.1],
                        "data_source": "test_source",
                        "development_contact_status": "none",
                        "contact_evidence": "synthetic no-contact evidence",
                        "expected_role": "positive_change_candidate",
                        "label_start_path": str(start_path),
                        "label_end_path": str(end_path),
                        "prediction_path": str(direct_path),
                        "flus_prediction_path": str(flus_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result_dir = tmp_path / "las_result"
    simulated_dir = result_dir / "simulated"
    comparison_dir = result_dir / "comparison_vs_flus"
    simulated_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    np.save(simulated_dir / "toy_area_2020_2021_paper58_las.npy", las)

    with (comparison_dir / "las_comparison_by_area.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "area",
                "start_year",
                "end_year",
                "tier",
                "stratum",
                "change_f1_advantage",
                "fom_advantage",
                "transition_accuracy_advantage",
                "allocation_disagreement_advantage",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "area": "toy_area",
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                "change_f1_advantage": 0.5,
                "fom_advantage": 0.25,
                "transition_accuracy_advantage": 0.5,
                "allocation_disagreement_advantage": -0.1,
            }
        )
    (comparison_dir / "las_comparison_summary.json").write_text(
        json.dumps(
            {
                "advantages": {
                    metric: {
                        "mean_advantage": value,
                        "bootstrap_ci": {"ci_low": value - 0.1, "ci_high": value + 0.1},
                    }
                    for metric, value in {
                        "change_f1": 0.5,
                        "fom": 0.25,
                        "transition_accuracy": 0.5,
                        "allocation_disagreement": -0.1,
                    }.items()
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "report"
    make_las_visual_report(registry, result_dir, output_dir)

    assert (output_dir / "README.md").stat().st_size > 0
    assert (output_dir / "report.html").stat().st_size > 0
    assert (output_dir / "figures" / "fig1_metric_advantages.png").stat().st_size > 0
    assert (output_dir / "figures" / "fig2_area_advantages.png").stat().st_size > 0
    assert (output_dir / "figures" / "spatial_toy_area.png").stat().st_size > 0
