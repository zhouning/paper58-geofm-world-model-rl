# Design Spec: Merged Manuscript `geofm_world_model_rl.tex`

**Date:** 2026-04-21
**Target journal:** Remote Sensing of Environment (RSE), IF ~13.5
**Output file:** `D:/test/geofm_world_model_rl.tex` (plus companion `.docx`)
**Source documents:**
- `D:/adk/docs/paper5_world_model_paper.tex` (World Model, 709 lines)
- `D:/test/paper8_design.md` (Paper 8 design notes)
- `D:/test/paper8/train_dual_rep.py` + `D:/test/paper8/dual_rep_env.py`
- `D:/test/paper8/results/dual_rep/` (15×3 = 45 eval JSONs, stats JSON)

## 1. Purpose and Framing

Produce a single RSE-style manuscript that integrates (a) the JEPA-style geospatial world model from Paper 5 with (b) the downstream embedding-space planning RL experiments from Paper 8. The **world model is the primary contribution**; the RL section demonstrates that the learned dynamics are useful for decision support beyond forecasting.

Narrative spine:
1. GeoFM embeddings are a viable state space for geospatial world modeling (Paper 5).
2. A 459K-parameter LatentDynamicsNet on frozen AlphaEarth embeddings outperforms persistence on 17 Chinese study areas (Paper 5).
3. The same embedding state space admits cross-region RL optimization via dual representation + feature dropout (Paper 8).
4. Feature channel is decision-critical; embedding channel provides transferability — concrete guidance for practitioners adopting GeoFM world models.

## 2. Out-of-Scope

- Retraining the world model or adding new study areas.
- Any new DRL seeds beyond the 45 already completed (full/0.3/1.0 × 15 seeds).
- Chinese `.docx` translation (will be a separate follow-up, analogous to prior papers).
- Any claim about scenario-conditioned counterfactuals — Paper 5's caveat (baseline-only training) stays.

## 3. Manuscript Structure

| # | Section | Primary source | Target pages |
|---|---------|----------------|--------------|
| 1 | Introduction | Paper 5 intro + 1 paragraph on downstream planning | 1.5 |
| 2 | Related Work | Paper 5 related work + new subsection "World models for decision making" | 1.5 |
| 3 | Study Areas and Data | Paper 5 verbatim (17 areas, AlphaEarth, DEM) + short note on Bishan county parcel data used in Section 6 | 1 |
| 4 | Methodology: Geospatial World Model | Paper 5 Section 4 verbatim (LatentDynamicsNet, L2 normalization, dilated conv, multi-step loss, decoder) | 4 |
| 5 | World Model Experiments | Paper 5 Section 5 (three-phase validation, ablation, multi-step rollout) | 3 |
| 6 | **Downstream Application: Embedding-Space Planning** (new) | Paper 8 condensed | 3 |
| 7 | Discussion | Merged: world model value + downstream evidence + limitations | 1.5 |
| 8 | Conclusion and Future Work | Merged | 0.5 |

Total target: ~16 pages main text (RSE typical range 10–20).

## 4. Section 6 Detailed Plan

### 6.1 Motivation (0.3 pp)
- Question: can a frozen-GeoFM world model support decision-making, not just forecasting?
- Connection to Section 5 (short: dynamics are well-characterized, now probe their decision utility).
- Positioning vs model-free baselines: we do not claim state-of-the-art planning performance; we probe whether the learned dynamics carry enough signal to drive a policy AND whether embedding-only input sustains performance for transfer.

### 6.2 Problem setup (0.5 pp)
- County-level farmland consolidation task on Bishan (same task family as Papers 2–4 in this project, but briefly re-explained so RSE readers do not need prior context).
- Action: pick one of N blocks per step to trigger a deterministic consolidation sub-routine.
- State per block: 17 engineered features (from cadastral parcels) + 64-dim GeoFM embedding.
- Reward: weighted improvement in slope-minimization and contiguity metrics (formal definitions in appendix).

### 6.3 Dual-representation environment (0.7 pp)
- Schematic + equations for `DualRepEnv`.
- 17-dim feature channel: fast, local, task-specific.
- 64-dim embedding channel: from Paper 5 world model encoder; region-transferable.
- MaskablePPO + ParcelScoringPolicy architecture (brief; details in appendix/code release).
- Feature dropout mechanism: probability *p* of zeroing the feature channel per step during training to simulate feature-unavailable target regions.

### 6.4 Experimental protocol (0.3 pp)
- 15 seeds per configuration (p = 0.0, 0.3, 1.0), 100k steps each.
- Evaluation on 5 episodes of the real CountyLevelEnv per trained model.
- Metrics: slope change %, contiguity change, training time.
- Statistical tests: independent t-test, Mann–Whitney U, Cohen's d (results file: `paper8/results/dual_rep/dropout_statistical_tests.json`).

### 6.5 Results (0.7 pp)
- Table with 3 rows × {slope mean±std, contiguity mean±std, n}.
- Pairwise significance matrix.
- Key numbers (verified from JSON files):
  - full (0.0): slope -0.742%±0.151%, cont +0.0152±0.0042
  - dropout 0.3: slope -0.644%±0.151%, cont +0.0146±0.0051
  - dropout 1.0: slope -0.096%±0.048%, cont +0.0064±0.0036
  - full vs 0.3: p=0.097 (n.s.), d=-0.63
  - full vs 1.0: p<0.0001, d=-5.57
  - 0.3 vs 1.0: p<0.0001, d=-4.71

### 6.6 Interpretation (0.5 pp)
- Feature channel is decision-critical (full-dropout collapses to ~13% of full performance).
- Moderate dropout (0.3) retains ~87% performance with no significant loss — acceptable price for partial feature absence at transfer.
- Cross-region transfer pathway: train with dropout on data-rich region, deploy to feature-poor region using embedding-only input; degradation bounded by the p=1.0 floor.
- Caveat: transfer to truly unseen regions is not evaluated here; the result is a robustness demonstration, not a transfer demonstration. Marked as future work.

## 5. Other Section Changes (relative to Paper 5)

### Introduction
- Add one paragraph after the contributions list: "Beyond forecasting, we probe whether the learned world model supports downstream decision making. Section 6 shows that policies trained on the dual representation of feature + embedding channels retain 87% of baseline performance under moderate feature dropout, providing guidance for cross-region deployment."
- Update contribution list to 5 items (original 4 + "downstream planning utility").

### Related Work
- Add subsection 2.4 "World models for decision making" (~150 words): Dreamer family, MuZero, learned-env planning in robotics; lack of geospatial counterpart.

### Study Areas and Data
- Append short paragraph: "For the downstream planning experiment (Section 6), we additionally use the cadastral parcel dataset for Bishan County (approx. 101,657 parcels aggregated into 2,600 management blocks), with per-block features computed from DEM-derived slope, land-use classes, and boundary geometry."

### Discussion
- Merge original Paper 5 discussion with three new paragraphs:
  - Why feature channel dominates (hypothesis: task-specific engineered features encode consolidation-relevant geometry that GeoFM embeddings lack).
  - When embedding-only input helps (transfer, reduced feature engineering burden).
  - Limitations: single task family, no held-out region transfer, scenario-conditioning still untrained (inherited from Paper 5).

### Conclusion and Future Work
- Append: training scenario-conditioned dynamics via policy interventions; cross-region zero-shot transfer on the 17 world-model regions.

## 6. Figures and Tables

Additions on top of Paper 5's figures:

- **Fig. 6.1**: DualRepEnv schematic (one TikZ diagram). Similar style to Fig. 1.
- **Fig. 6.2**: Box plot of slope change across 3 configurations, 15 seeds each.
- **Table 6.1**: Summary statistics (3 rows).
- **Table 6.2**: Pairwise significance matrix.

All existing Paper 5 figures retained.

## 7. References

Preserve Paper 5's bib entries. Add:
- Paper 7 (if available): learned environment for RL in this project.
- PPO / MaskablePPO (Schulman 2017; Huang 2022).
- Prior RL-on-world-model works (Ha & Schmidhuber 2018 already present; add Hafner Dreamer V3 if not already cited — it already is).

No fabricated references — every new citation must be verified against an existing paper's references or a verified DOI.

## 8. Acceptance Criteria

- `geofm_world_model_rl.tex` compiles to PDF without errors.
- Word count ≈ 7,500–9,500 (RSE typical).
- All numbers in Section 6 exactly match the source JSON files.
- Paper 5's original content preserved byte-identically wherever unchanged (not paraphrased).
- New Section 6 + associated figures/tables present.
- No `TBD`, `TODO`, or placeholder strings.
- Cross-references (Fig./Tab./Sec./Eq.) resolve.
- `D:/adk/docs/paper5_world_model_paper.tex` not modified.

## 9. Implementation Plan (handoff to writing-plans skill)

1. Copy Paper 5 tex to `D:/test/geofm_world_model_rl.tex`.
2. Update title block (keep authorship scheme).
3. Edit Introduction (add downstream paragraph + 5th contribution).
4. Edit Related Work (add subsection 2.4).
5. Edit Data section (add Bishan paragraph).
6. Insert new Section 6 between Section 5 and current Discussion.
7. Extend Discussion with 3 new paragraphs; keep original Paper 5 text intact above them.
8. Extend Conclusion/Future Work.
9. Add bib entries (PPO, MaskablePPO, Paper 7 if cited).
10. Compile with `pdflatex` + `bibtex`; fix any undefined refs.
11. Update `MEMORY.md` with manuscript status.
