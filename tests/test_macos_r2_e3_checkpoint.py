from pathlib import Path
import importlib.util
import io
import math


def _load_e3_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / "e3_multistep_all_areas.py"
    spec = importlib.util.spec_from_file_location("e3_multistep_all_areas", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_ckpt_prefers_ae_ckpt_env(monkeypatch, tmp_path):
    module = _load_e3_module()
    ckpt = tmp_path / "custom_latent_dynamics_v1.pt"
    ckpt.write_bytes(b"checkpoint")

    monkeypatch.setenv("AE_CKPT", str(ckpt))

    assert module.resolve_ckpt() == ckpt


def test_checkpoint_candidates_include_repo_adk_weight():
    module = _load_e3_module()
    expected = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "adk_world_model"
        / "weights"
        / "latent_dynamics_v1.pt"
    )

    assert expected in module.AE_CKPT_CANDIDATES


def test_e3_summarizes_rows_by_step():
    module = _load_e3_module()
    rows = [
        {"area": "a", "step": 1, "persistence": 0.9, "model": 0.8, "advantage": -0.1},
        {"area": "b", "step": 1, "persistence": 0.8, "model": 0.9, "advantage": 0.1},
        {"area": "a", "step": 2, "persistence": 0.7, "model": 0.6, "advantage": -0.1},
    ]

    summary, tests = module.summarize_multistep_rows(rows)

    assert summary["steps"]["1"]["n"] == 2
    assert math.isclose(summary["steps"]["1"]["mean_advantage"], 0.0, abs_tol=1e-12)
    assert summary["steps"]["1"]["n_pos"] == 1
    assert summary["steps"]["1"]["n_neg"] == 1
    assert tests["steps"]["1"]["n"] == 2
    assert "wilcoxon_p" in tests["steps"]["1"]
    assert summary["steps"]["2"]["n"] == 1


def test_e3_csv_writer_uses_lf_line_endings():
    module = _load_e3_module()
    out = io.StringIO()
    writer = module.make_csv_writer(out, fieldnames=["area", "step"])

    writer.writeheader()
    writer.writerow({"area": "a", "step": 1})

    assert "\r" not in out.getvalue()
    assert out.getvalue().splitlines() == ["area,step", "a,1"]


def test_e3_repo_rel_records_paths_relative_to_repository():
    module = _load_e3_module()
    path = module.REPO_ROOT / "experiments" / "macos_r2" / "weights" / "retrain_v2" / "latent_dynamics_v2_seed456.pt"

    assert module.repo_rel(path) == "experiments/macos_r2/weights/retrain_v2/latent_dynamics_v2_seed456.pt"
