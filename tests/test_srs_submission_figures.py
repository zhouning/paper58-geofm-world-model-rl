import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRS_DIR = ROOT / "paper" / "srs_submission_paper58"
MANUSCRIPT = SRS_DIR / "manuscript" / "srs_geofm_embedding_change_screening.tex"


def _referenced_figure_files() -> set[str]:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    matches = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", text)
    return {Path(match).name for match in matches}


def test_srs_manuscript_does_not_reference_obsolete_encoder_diagnostic_figure():
    referenced = _referenced_figure_files()

    assert "fig_encoder_diagnostic.pdf" not in referenced


def test_srs_figure_directories_only_contain_current_referenced_figures():
    referenced = _referenced_figure_files()
    expected = set(referenced)
    expected.update(Path(name).with_suffix(".png").name for name in referenced)

    for figure_dir in (SRS_DIR / "figures", SRS_DIR / "manuscript" / "figures"):
        actual = {path.name for path in figure_dir.iterdir() if path.is_file()}
        assert actual == expected
