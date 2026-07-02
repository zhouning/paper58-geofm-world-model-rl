from pathlib import Path
import importlib.util

import numpy as np


def _load_module(filename: str):
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e4_finds_lulc_label_filename(monkeypatch, tmp_path):
    module = _load_module("e4_per_year_decoder.py")
    labels = tmp_path / "labels"
    labels.mkdir()
    expected = labels / "sample_area_lulc_2020.npy"
    np.save(expected, np.array([[1]], dtype=np.uint8))

    monkeypatch.setattr(module, "INDEP_LABELS", labels)

    assert module.find_label_path("sample_area", 2020) == expected


def test_e5_discovers_lulc_prediction_pairs(monkeypatch, tmp_path):
    module = _load_module("e5_sa_alloc_sensitivity.py")
    pred = tmp_path / "predicted"
    pred.mkdir()
    np.save(pred / "sample_area_lulc_pred_2020_2021.npy", np.array([[2]], dtype=np.uint8))

    monkeypatch.setattr(module, "INDEP_PRED", pred)

    assert module.load_v2_registry() == [
        {"area": "sample_area", "start_year": 2020, "end_year": 2021}
    ]


def test_e5_loads_lulc_pair_files(monkeypatch, tmp_path):
    module = _load_module("e5_sa_alloc_sensitivity.py")
    labels = tmp_path / "labels"
    pred = tmp_path / "predicted"
    labels.mkdir()
    pred.mkdir()
    np.save(labels / "sample_area_lulc_2020.npy", np.array([[1]], dtype=np.uint8))
    np.save(labels / "sample_area_lulc_2021.npy", np.array([[2]], dtype=np.uint8))
    np.save(pred / "sample_area_lulc_pred_2020_2021.npy", np.array([[2]], dtype=np.uint8))

    monkeypatch.setattr(module, "INDEP_LABELS", labels)
    monkeypatch.setattr(module, "INDEP_PRED", pred)

    data = module.load_township_pair("sample_area", 2020, 2021)

    assert data is not None
    assert data["start_lab"].tolist() == [[1]]
    assert data["end_lab"].tolist() == [[2]]
    assert data["pred_lab"].tolist() == [[2]]


def test_e4_embedding_files_for_year_uses_multiple_roots_without_duplicates(monkeypatch, tmp_path):
    module = _load_module("e4_per_year_decoder.py")
    primary = tmp_path / "independent_embeddings"
    secondary = tmp_path / "paper8_data"
    primary.mkdir()
    secondary.mkdir()
    (primary / "holdout_emb_2020.npy").touch()
    (primary / "bishan_emb_2020.npy").touch()
    (secondary / "bishan_emb_2020.npy").touch()
    (secondary / "paper8_only_emb_2020.npy").touch()

    monkeypatch.setattr(module, "AE_DIRS", [primary, secondary])

    assert module.embedding_files_for_year(2020) == [
        ("bishan", primary / "bishan_emb_2020.npy"),
        ("holdout", primary / "holdout_emb_2020.npy"),
        ("paper8_only", secondary / "paper8_only_emb_2020.npy"),
    ]
