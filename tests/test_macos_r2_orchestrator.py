from pathlib import Path
import importlib.util
import shutil
import subprocess


def _load_run_markers():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments" / "macos_r2" / "run_markers.py"
    spec = importlib.util.spec_from_file_location("run_markers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_done_marker_does_not_create_canonical_done(tmp_path):
    markers = _load_run_markers()

    written = markers.mark_done(tmp_path, smoke=True)

    assert written == tmp_path / ".done_smoke"
    assert written.exists()
    assert not (tmp_path / ".done").exists()
    assert not (tmp_path / ".done_full").exists()


def test_full_done_marker_creates_canonical_and_full_done(tmp_path):
    markers = _load_run_markers()
    (tmp_path / ".done_smoke").touch()

    written = markers.mark_done(tmp_path, smoke=False)

    assert written == tmp_path / ".done_full"
    assert (tmp_path / ".done").exists()
    assert (tmp_path / ".done_full").exists()
    assert not (tmp_path / ".done_smoke").exists()


def test_orchestrator_runs_under_macos_bash_for_empty_selection():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "macos_r2" / "run_all_macos.sh"

    result = subprocess.run(
        ["/bin/bash", str(script), "--only", "not_an_experiment", "--smoke"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "selected:" in result.stdout
    assert "run summary JSON:" in result.stdout


def test_orchestrator_does_not_repeat_script_name_for_no_arg_experiment(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "macos_r2" / "run_all_macos.sh"
    fake_python = script.parent / "python"
    fake_python.write_text("#!/bin/sh\necho \"$@\"\n")
    fake_python.chmod(0o755)

    monkeypatch.setenv("PATH", f"{script.parent}:{Path('/usr/bin')}:{Path('/bin')}")
    try:
        result = subprocess.run(
            ["/bin/bash", str(script), "--only", "e3", "--smoke"],
            cwd=script.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        fake_python.unlink()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "running python e3_multistep_all_areas.py --smoke" in result.stdout
    assert "e3_multistep_all_areas.py e3_multistep_all_areas.py" not in result.stdout


def test_resume_ignores_legacy_smoke_done_without_full_marker(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "macos_r2" / "run_all_macos.sh"
    fake_python = script.parent / "python"
    fake_python.write_text("#!/bin/sh\necho \"$@\"\n")
    fake_python.chmod(0o755)
    smoke_result = script.parent / "results" / "e7_legacy_smoke_marker"
    smoke_result.mkdir(parents=True, exist_ok=True)
    (smoke_result / ".done").touch()
    (smoke_result / ".done_smoke").touch()

    monkeypatch.setenv("PATH", f"{script.parent}:{Path('/usr/bin')}:{Path('/bin')}")
    try:
        result = subprocess.run(
            ["/bin/bash", str(script), "--resume", "--smoke"],
            cwd=script.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        fake_python.unlink()
        shutil.rmtree(smoke_result)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "running python e7_third_encoder.py --smoke" in result.stdout
    assert "[e7_third_encoder] SKIP (already full .done)" not in result.stdout
