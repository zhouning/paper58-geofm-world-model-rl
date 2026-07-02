from pathlib import Path
import importlib.util


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
