from pathlib import Path
import importlib.util
import json

import pandas as pd


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / "extract_v4_numbers.py"
    spec = importlib.util.spec_from_file_location("extract_v4_numbers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_e3_result_is_json_serializable(monkeypatch, tmp_path):
    module = _load_module()
    results = tmp_path / "results"
    out = results / "e3_multistep"
    out.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "area": "a",
                "step": 1,
                "persistence": 0.99,
                "model": 0.98,
                "advantage": -0.01,
            }
        ]
    ).to_csv(out / "multistep_all_areas.csv", index=False)
    monkeypatch.setattr(module, "RESULTS_DIR", results)

    extracted = module.extract_e3()

    assert extracted["bug_fixed"] is True
    json.dumps(extracted)


def test_extract_e4_reads_current_decoder_accuracy_column(monkeypatch, tmp_path):
    module = _load_module()
    results = tmp_path / "results"
    out = results / "e4_per_year_decoder"
    out.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "year": 2024,
                "n_samples": 800,
                "cv_acc_mean": 0.7475,
                "cv_macro_f1_mean": 0.4617,
            }
        ]
    ).to_csv(out / "decoder_by_year.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": "sample_2023-2024",
                "end_year": 2024,
                "v2_end_accuracy": 0.59,
                "retrained_end_accuracy": 0.75,
                "delta": 0.16,
                "status": "ok",
            }
        ]
    ).to_csv(out / "per_pair_end_accuracy_delta.csv", index=False)
    monkeypatch.setattr(module, "RESULTS_DIR", results)

    extracted = module.extract_e4()

    assert extracted["per_year"][2024]["cv_accuracy"] == 0.7475
    json.dumps(extracted)
