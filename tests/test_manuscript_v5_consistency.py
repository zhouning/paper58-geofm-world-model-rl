from scripts.rse_revision.check_manuscript_v5_consistency import run_checks, _check_submission_context, _check_reference_heading


def test_manuscript_v5_consistency_checks_pass_on_current_sources():
    report = run_checks()

    assert report.errors == []
    assert report.abstract_words <= 250
    assert report.total_words <= 15000
    assert 3 <= report.highlights_count <= 5
    assert report.highlight_max_chars <= 85
    assert report.table_rows_checked >= 10

def test_submission_context_check_rejects_internal_revision_language():
    errors = _check_submission_context(
        "Reviewer issue S6 asked for a prior v3 revision package; "
        "the R1 baseline used a pre-R2 checkpoint at commit 9c2ba1d. "
        "The revised manuscript keeps a revision audit script and a current manuscript note. "
        "The reviewer correctly pointed out issue S2. "
        "The pre-retrain baseline and legacy aggregate table are retained for provenance only. "
        "We now use a planning task as a bounded application probe."
    )

    assert any("Reviewer issue" in error for error in errors)
    assert any("prior v3 revision package" in error for error in errors)
    assert any("pre-R2 checkpoint" in error for error in errors)
    assert any("commit" in error for error in errors)
    assert any("revised manuscript" in error for error in errors)
    assert any("revision audit script" in error for error in errors)
    assert any("current manuscript" in error for error in errors)
    assert any("reviewer" in error.lower() for error in errors)
    assert any("issue S" in error for error in errors)
    assert any("pre-retrain baseline" in error for error in errors)
    assert any("legacy aggregate" in error for error in errors)
    assert any("retained for provenance" in error for error in errors)
    assert any("We now" in error for error in errors)

def test_reference_heading_check_rejects_manual_references_section_before_bibliography():
    errors = _check_reference_heading(
        "\\section*{References}\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{key} Example.\n"
        "\\end{thebibliography}\n"
    )

    assert any("manual References" in error for error in errors)

