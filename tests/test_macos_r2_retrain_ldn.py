from pathlib import Path
import importlib.util
import json
import io

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / "retrain_ldn_on_r2_data.py"
    spec = importlib.util.spec_from_file_location("retrain_ldn_on_r2_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_stats_include_full_inference_fields():
    module = _load_module()

    stats = module.paired_stats_from_advantages(
        np.array([-0.02, 0.01, 0.03, -0.01], dtype=np.float32),
        best_ckpt="latent_dynamics_v2_seed456.pt",
        n_perm=1000,
        n_boot=500,
    )

    assert stats["n"] == 4
    assert stats["n_pos"] == 2
    assert stats["n_neg"] == 2
    assert stats["best_ckpt"] == "latent_dynamics_v2_seed456.pt"
    for key in ["permutation_p", "bootstrap_ci_lo", "bootstrap_ci_hi", "cohen_dz"]:
        assert key in stats
    json.dumps(stats)


def test_repo_rel_records_paths_relative_to_repository():
    module = _load_module()

    path = module.HERE / "weights" / "retrain_v2" / "latent_dynamics_v2_seed42.pt"

    assert module.repo_rel(path) == "experiments/macos_r2/weights/retrain_v2/latent_dynamics_v2_seed42.pt"


def test_csv_writer_uses_lf_line_endings():
    module = _load_module()
    out = io.StringIO()
    writer = module.make_csv_writer(out, fieldnames=["area", "advantage"])

    writer.writeheader()
    writer.writerow({"area": "a", "advantage": 0.1})

    assert "\r" not in out.getvalue()
    assert out.getvalue().splitlines() == ["area,advantage", "a,0.1"]
