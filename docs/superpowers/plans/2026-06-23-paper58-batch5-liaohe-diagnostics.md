# Paper58 Batch 5 Liaohe Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Batch 5 Liaohe diagnostic readout that explains the negative wetland row while preserving Batch 2 as contradictory evidence.

**Architecture:** Reuse the existing Batch 2 diagnostic primitives for alignment, transition fate, decoder confidence, and forecast confidence. Add one thin Batch 5 orchestration script that writes Batch 5-specific aggregate outputs and summaries.

**Tech Stack:** Python, NumPy, existing Paper58 benchmark CSV/JSON outputs, existing pytest suite.

---

### Task 1: Batch 5 Diagnostic Tests

**Files:**
- Create: `tests/test_paper58_batch5_liaohe_diagnostics.py`
- Modify: `tests/test_paper58_batch2_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Add tests that expect a new `build_liaohe_diagnostics` orchestration function, Batch 5-specific output filenames, and optional helper output filenames for reused Batch 2 diagnostics.

- [ ] **Step 2: Run the focused tests**

Run:

```text
python -m pytest tests/test_paper58_batch5_liaohe_diagnostics.py tests/test_paper58_batch2_diagnostics.py -q
```

Expected before implementation: failure because the Batch 5 module and helper filename option do not exist yet.

### Task 2: Batch 5 Diagnostic Implementation

**Files:**
- Create: `scripts/paper58_benchmark/make_batch5_liaohe_diagnostics.py`
- Modify: `scripts/paper58_benchmark/make_batch2_diagnostics.py`

- [ ] **Step 1: Add optional output filenames to reusable helpers**

Keep existing defaults unchanged while allowing Batch 5 callers to write non-`batch2_*` filenames.

- [ ] **Step 2: Add the Batch 5 orchestration script**

The script reads `benchmark_results_batch5/benchmark_metrics_by_pair.csv`, runs diagnostics for Liaohe, prior wetland comparison rows, and WEnan, writes aggregate CSV outputs, and writes JSON/text summaries.

- [ ] **Step 3: Run focused tests**

Run the same focused pytest command and require all tests to pass.

### Task 3: Real Diagnostic Run And Handoff Update

**Files:**
- Create outputs under `paper/rse_submission_paper58/diagnostics_batch5_liaohe`
- Modify: `docs/current_work_progress_2026-06-20.md`

- [ ] **Step 1: Run the diagnostic script**

Run:

```text
python -m scripts.paper58_benchmark.make_batch5_liaohe_diagnostics
```

- [ ] **Step 2: Read the summary outputs**

Extract the Liaohe primary/spatial metrics, best-shift result, dominant transition fate, observed-vs-forecast confidence pattern, and WEnan weak-primary risk.

- [ ] **Step 3: Update the handoff document**

Append the Batch 5 Liaohe diagnostic results while preserving the rule that Batch 2 failure remains visible.

- [ ] **Step 4: Verify and commit**

Run focused tests, the diagnostic script, and `git diff --check`; then commit the code, outputs, and handoff update.
