from pathlib import Path
import importlib.util

import numpy as np
import pytest
import torch


def _load_module(filename: str):
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e1_extracts_tokens_from_prithvi_tuple_output():
    module = _load_module("e1_prithvi_patch_extract.py")
    tokens = torch.ones((2, 64, 768))

    assert module.spatial_tokens_from_prithvi_output((tokens, (8, 8))) is tokens


def test_e1_accepts_direct_prithvi_tensor_output():
    module = _load_module("e1_prithvi_patch_extract.py")
    tokens = torch.ones((2, 64, 768))

    assert module.spatial_tokens_from_prithvi_output(tokens) is tokens


def test_e1_smoke_extracts_two_consecutive_years():
    module = _load_module("e1_prithvi_patch_extract.py")

    assert module.SMOKE_YEARS == [2020, 2021]


def test_e1_formats_single_area_paired_stats_without_missing_float_crash():
    module = _load_module("e1_prithvi_patch_train_eval.py")

    assert module.format_paired_stats({"n": 1}) == "n=1 mean=NA wilcoxon_p=NA"


def test_e6_resolve_ckpt_finds_repo_adk_weight(monkeypatch):
    module = _load_module("e6_expand_areas.py")
    repo = Path(__file__).resolve().parents[1]
    expected = repo / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt"

    monkeypatch.delenv("AE_CKPT", raising=False)

    assert module.resolve_ckpt() == expected


def test_e6_discovers_eval_area_sources_from_multiple_roots(tmp_path):
    module = _load_module("e6_expand_areas.py")
    primary = tmp_path / "paper8_data"
    secondary = tmp_path / "independent_embeddings"
    primary.mkdir()
    secondary.mkdir()
    for year in [2017, 2018, 2019]:
        (primary / f"bishan_emb_{year}.npy").touch()
    for year in [2020, 2021]:
        (secondary / f"holdout_area_emb_{year}.npy").touch()
        (secondary / f"bishan_emb_{year}.npy").touch()
    (secondary / "single_year_area_emb_2020.npy").touch()

    sources = module.discover_eval_area_sources([primary, secondary])

    assert sources["bishan"]["root"] == primary
    assert sources["bishan"]["years"] == [2017, 2018, 2019]
    assert sources["holdout_area"]["root"] == secondary
    assert sources["holdout_area"]["years"] == [2020, 2021]
    assert "single_year_area" not in sources


def test_e6_summarizes_eval_area_sources_by_root(tmp_path):
    module = _load_module("e6_expand_areas.py")
    primary = tmp_path / "paper8_data"
    secondary = tmp_path / "independent_embeddings"

    summary = module.summarize_eval_area_sources({
        "bishan": {"root": primary, "years": [2017, 2018, 2019]},
        "holdout_area": {"root": secondary, "years": [2020, 2021]},
        "other_holdout": {"root": secondary, "years": [2020, 2021]},
    })

    assert summary["n_areas"] == 3
    assert summary["roots"][str(primary)] == 1
    assert summary["roots"][str(secondary)] == 2


def test_e6_full_extraction_backfills_study_areas_before_new_areas(monkeypatch):
    module = _load_module("e6_expand_areas.py")

    monkeypatch.setattr(module, "study_area_bboxes", lambda: [("study_area", [0, 0, 1, 1])])
    monkeypatch.setattr(module, "NEW_AREAS", [("new_area", [1, 1, 2, 2])])

    assert module.extraction_areas(smoke=False) == [
        ("study_area", [0, 0, 1, 1]),
        ("new_area", [1, 1, 2, 2]),
    ]
    assert module.extraction_areas(smoke=True) == [("new_area", [1, 1, 2, 2])]


def test_e6_cmd_extract_raises_when_any_area_year_fails(monkeypatch, tmp_path):
    module = _load_module("e6_expand_areas.py")

    monkeypatch.setattr(module, "NEW_AREAS", [("bad_area", [0, 0, 0.1, 0.1])])
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "init_gee", lambda: True)
    monkeypatch.setattr(module, "extract_embedding_grid", lambda bbox, year: None)
    monkeypatch.setattr(module, "extract_terrain", lambda bbox, target_shape=None: np.zeros((2, 1, 1), dtype=np.float32))

    with pytest.raises(SystemExit):
        module.cmd_extract(smoke=True)
