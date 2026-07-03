from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT = ROOT / "paper" / "rse_submission_paper58" / "manuscript" / "rse_geofm_world_model_rl_v5.tex"
DEFAULT_INDEPENDENT_CSV = (
    ROOT / "paper" / "rse_submission_paper58" / "revision_results_v2" / "independent_change_paired_tests.csv"
)
DEFAULT_HIGHLIGHTS = ROOT / "paper" / "rse_submission_paper58" / "submission_docs" / "highlights.md"
MAX_RSE_RESEARCH_WORDS = 15000
MAX_ABSTRACT_WORDS = 250
MAX_HIGHLIGHT_CHARS = 85
MIN_HIGHLIGHTS = 3
MAX_HIGHLIGHTS = 5

BANNED_PHRASES = [
    "authorship pending",
    "KG-VSF",
    "Prithvi-EO-2.0 100M",
    "2023\u95c2?024",
    "6 wins / 5 losses, Wilcoxon $p=0.72$",
    "stable overall advantage",
    "LAS achieves aggregate advantages over GeoSOS-FLUS",
    "weeks to minutes",
    "driver-free LULC simulation",
]

REQUIRED_SNIPPETS = [
    "flooded vegetation is folded into water",
    "snow/ice and bare ground are merged into ``bare/snow,''",
    "clouds are dropped",
    "The public repository is available as a \\href{https://github.com/zhouning/paper58-geofm-world-model-rl}{GitHub repository}",
    "Ravirathinam, P. et al., 2024. Towards a Knowledge guided Multimodal Foundation Model",
    "Chen, Y. et al., 2025. RemoteBAGEL: Remote Sensing-Oriented World Model",
    "Jakubik, J. et al., 2023. Foundation Models for Generalist Geospatial Artificial Intelligence",
    "Huang, S. and Onta\\~{n}\\'{o}n, S., 2020. A Closer Look at Invalid Action Masking",
]

METRIC_LABELS = {
    "change_f1": "Change F1",
    "end_accuracy": "End-year accuracy",
    "changed_pixel_accuracy": "Changed-pixel accuracy",
    "area_bias_mae": "Area-bias MAE (lower)",
    "transition_exact_match": "Transition exact match",
}

CONTROL_LABELS = {
    "shuffled": "Shuffled model",
    "prior": "Transition prior",
    "persistence": "Persistence",
}

LOWER_IS_BETTER = {"area_bias_mae"}


@dataclass
class CheckReport:
    errors: list[str]
    abstract_words: int
    total_words: int
    highlights_count: int
    highlight_max_chars: int
    table_rows_checked: int
    cited_keys: int
    bibitems: int

    @property
    def ok(self) -> bool:
        return not self.errors


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _latex_to_words(text: str) -> list[str]:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r" \1 ", text)
    text = text.replace("--", " ")
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)


def _abstract_word_count(tex: str) -> int:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    if not match:
        raise ValueError("abstract environment not found")
    return len(_latex_to_words(match.group(1)))


def _manuscript_word_count(tex: str) -> int:
    start_match = re.search(r"\\begin\{abstract\}", tex)
    end_match = re.search(r"\\end\{thebibliography\}", tex)
    if not start_match or not end_match:
        raise ValueError("cannot locate countable manuscript span")
    countable = tex[start_match.start() : end_match.end()]
    return len(_latex_to_words(countable))


def _load_highlights(path: Path) -> list[str]:
    if not path.exists():
        return []
    highlights: list[str] = []
    for line in _read(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            highlights.append(stripped[2:].strip())
    return highlights

def _strip_nonprose_latex(tex: str) -> str:
    tex = re.sub(r"\\path\{[^{}]*\}", " ", tex)
    tex = re.sub(r"\\url\{[^{}]*\}", " ", tex)
    tex = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", tex)
    tex = re.sub(r"^\s*\\usepackage(?:\[[^\]]*\])?\{[^{}]*\}\s*$", " ", tex, flags=re.M)
    return tex


SUBMISSION_CONTEXT_PATTERNS = [
    r"Reviewer issue",
    r"reviewer request",
    r"reviewer-requested",
    r"\breviewer\b",
    r"\bissue S\d+\b",
    r"revised manuscript",
    r"current manuscript",
    r"prior v\d+ revision package",
    r"v\d+ integration",
    r"earlier revision",
    r"prior revision",
    r"R1 baseline",
    r"R2 revision",
    r"pre-R2 checkpoint",
    r"pre-retrain baseline",
    r"pre-commit embeddings",
    r"initial repository commit",
    r"commit (?:\\texttt\{)?[0-9a-f]{7,40}\}?",
    r"revision audit(?: script)?",
    r"revision artefacts",
    r"revision artifacts",
    r"revision-results",
    r"revision-artefacts",
    r"legacy[^.\n]{0,80}aggregate",
    r"retained for provenance",
    r"unarchived embeddings",
    r"earlier[^.\n]{0,80}baseline",
    r"\bWe now\b",
]


def _check_submission_context(tex: str) -> list[str]:
    prose = _strip_nonprose_latex(tex)
    errors: list[str] = []
    for pattern in SUBMISSION_CONTEXT_PATTERNS:
        match = re.search(pattern, prose, flags=re.I)
        if match:
            errors.append(f"submission-context phrase should be rewritten for initial submission: {match.group(0)}")
    return errors


def _check_reference_heading(tex: str) -> list[str]:
    if re.search(r"\\section\*\{References\}", tex):
        return [
            "manual References heading found; thebibliography already creates the References heading"
        ]
    return []


def _strip_cell(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace(r"\,", "")
    cell = cell.replace("$", "")
    cell = cell.replace(r"\\", "")
    return re.sub(r"\s+", " ", cell).strip()


def _format_float(value: float, signed: bool = False) -> str:
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _load_independent_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(row["metric"], row["control"]): row for row in rows}


def _parse_independent_table(tex: str) -> dict[tuple[str, str], dict[str, str]]:
    label_pos = tex.find(r"\label{tab:independent_change_paired}")
    if label_pos < 0:
        raise ValueError("independent paired table label not found")
    table_text = tex[label_pos : tex.find(r"\end{table}", label_pos)]
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for raw_line in table_text.splitlines():
        if "&" not in raw_line or r"\\" not in raw_line:
            continue
        cells = [_strip_cell(cell) for cell in raw_line.split("&")]
        if len(cells) != 8 or cells[0] == "Metric":
            continue
        metric, control, model_mean, ctrl_mean, mean_delta, wins, wilcoxon, sign_p = cells
        rows[(metric, control)] = {
            "model_mean": model_mean,
            "ctrl_mean": ctrl_mean,
            "mean_delta": mean_delta,
            "wins": wins,
            "wilcoxon_p": wilcoxon,
            "sign_test_p": sign_p,
        }
    return rows


def _check_independent_table(tex: str, csv_path: Path) -> list[str]:
    errors: list[str] = []
    source = _load_independent_csv(csv_path)
    table = _parse_independent_table(tex)
    for (metric_label, control_label), actual in table.items():
        metric = next((key for key, label in METRIC_LABELS.items() if label == metric_label), None)
        control = next((key for key, label in CONTROL_LABELS.items() if label == control_label), None)
        if metric is None or control is None:
            errors.append(f"unknown independent-table row: {metric_label} / {control_label}")
            continue
        row = source.get((metric, control))
        if row is None:
            errors.append(f"table row not found in CSV: {metric} / {control}")
            continue
        if metric in LOWER_IS_BETTER:
            expected_wins = f"{int(row['n_neg'])} / {int(row['n_pos'])}"
        else:
            expected_wins = f"{int(row['n_pos'])} / {int(row['n_neg'])}"
        expected = {
            "model_mean": _format_float(float(row["model_mean"])),
            "ctrl_mean": _format_float(float(row["ctrl_mean"])),
            "mean_delta": _format_float(float(row["mean_diff"]), signed=True),
            "wins": expected_wins,
            "wilcoxon_p": _format_float(float(row["wilcoxon_p"])),
            "sign_test_p": _format_float(float(row["sign_test_p"])),
        }
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                errors.append(
                    f"independent table mismatch for {metric}/{control} {key}: "
                    f"manuscript={actual[key]} csv={expected_value}"
                )
    required_rows = {
        ("Change F1", "Shuffled model"),
        ("Changed-pixel accuracy", "Shuffled model"),
        ("Area-bias MAE (lower)", "Persistence"),
        ("Transition exact match", "Transition prior"),
    }
    missing = sorted(required_rows.difference(table))
    for metric_label, control_label in missing:
        errors.append(f"required independent-table row missing: {metric_label} / {control_label}")
    return errors


def _extract_citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"\\cite(?:p|t|alt|alp|author|year)?\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}")
    for match in pattern.finditer(tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _extract_bibitems(tex: str) -> set[str]:
    return set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^{}]+)\}", tex))


def run_checks(
    manuscript_path: Path = DEFAULT_MANUSCRIPT,
    independent_csv: Path = DEFAULT_INDEPENDENT_CSV,
    highlights_path: Path = DEFAULT_HIGHLIGHTS,
) -> CheckReport:
    tex = _read(manuscript_path)
    errors: list[str] = []

    abstract_words = _abstract_word_count(tex)
    if abstract_words > MAX_ABSTRACT_WORDS:
        errors.append(f"abstract too long: {abstract_words} words > {MAX_ABSTRACT_WORDS}")

    total_words = _manuscript_word_count(tex)
    if total_words > MAX_RSE_RESEARCH_WORDS:
        errors.append(f"manuscript too long for RSE research article: {total_words} words > {MAX_RSE_RESEARCH_WORDS}")

    highlights = _load_highlights(highlights_path)
    highlights_count = len(highlights)
    highlight_max_chars = max((len(item) for item in highlights), default=0)
    if not (MIN_HIGHLIGHTS <= highlights_count <= MAX_HIGHLIGHTS):
        errors.append(
            f"RSE highlights count out of range: {highlights_count} "
            f"not in [{MIN_HIGHLIGHTS}, {MAX_HIGHLIGHTS}]"
        )
    for index, highlight in enumerate(highlights, start=1):
        if len(highlight) > MAX_HIGHLIGHT_CHARS:
            errors.append(
                f"RSE highlight {index} too long: {len(highlight)} characters > {MAX_HIGHLIGHT_CHARS}"
            )

    for phrase in BANNED_PHRASES:
        if phrase in tex:
            errors.append(f"banned high-risk phrase still present: {phrase}")

    errors.extend(_check_submission_context(tex))
    errors.extend(_check_reference_heading(tex))

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in tex:
            errors.append(f"required manuscript/reference snippet missing: {snippet}")

    errors.extend(_check_independent_table(tex, independent_csv))

    cited = _extract_citation_keys(tex)
    bibitems = _extract_bibitems(tex)
    undefined = sorted(cited.difference(bibitems))
    uncited = sorted(bibitems.difference(cited))
    if undefined:
        errors.append("undefined citation keys: " + ", ".join(undefined))
    if uncited:
        errors.append("uncited bibliography items: " + ", ".join(uncited))

    if "8 wins / 3 losses, Wilcoxon $p=0.365$" not in tex:
        errors.append("changed-pixel explanatory bullet does not report 8 / 3 and p=0.365")

    return CheckReport(
        errors=errors,
        abstract_words=abstract_words,
        total_words=total_words,
        highlights_count=highlights_count,
        highlight_max_chars=highlight_max_chars,
        table_rows_checked=len(_parse_independent_table(tex)),
        cited_keys=len(cited),
        bibitems=len(bibitems),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check manuscript v5 against reviewer-risk consistency rules.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--independent-csv", type=Path, default=DEFAULT_INDEPENDENT_CSV)
    parser.add_argument("--highlights", type=Path, default=DEFAULT_HIGHLIGHTS)
    args = parser.parse_args()
    report = run_checks(args.manuscript, args.independent_csv, args.highlights)
    if report.ok:
        print(
            "OK: manuscript consistency checks passed "
            f"(abstract={report.abstract_words} words, "
            f"total={report.total_words} words, "
            f"highlights={report.highlights_count}, "
            f"max_highlight_chars={report.highlight_max_chars}, "
            f"independent_table_rows={report.table_rows_checked}, "
            f"cited_keys={report.cited_keys}, bibitems={report.bibitems})."
        )
        return
    print("Manuscript consistency checks failed:")
    for error in report.errors:
        print(f"- {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

