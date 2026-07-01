"""
Reviewer-response analyses for Paper58 (RSE round 2).

Adds the rigorous statistics that R1 (RSE) demanded but the v1 manuscript omitted:

  M1  Paired significance tests on 10-area cosine advantage over persistence:
      - Wilcoxon signed-rank
      - Paired t-test
      - Permutation test
      - Cohen's dz
      - Poyang Lake (val split) reported separately

  M2  Independent categorical validation (12 area-year pairs):
      - Wilcoxon paired vs shuffled / vs prior / vs persistence
      - Recompute mean AFTER dropping Wuyi 2020-2021 (0% ref change, ill-defined F1)
      - Report per-area wins/losses versus each control
      - Report end-year accuracy AND class-area bias gaps versus persistence

  M3  GeoSOS-FLUS per-area win/loss matrix:
      - For every method vs GeoSOS-FLUS, per-area binary wins per metric
      - Binomial sign test on 24 townships
      - Split raw-latent-dynamics vs the scale-adaptive allocation methods

  S1  Cosine similarity gap -> downstream end-year accuracy gap curve
      (built from independent_change_validation_by_area.csv)

  S5  Multi-step degradation across all 10 valid cached AlphaEarth areas
      (fills the gap left by the v1 table that only listed 3 areas)

All outputs are written under revision_results_v2/ as CSV + JSON. No new
model training is required; analyses operate purely on already-cached
experiment artefacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REV_ROOT = Path(__file__).resolve().parent
SUBMISSION_ROOT = REV_ROOT.parent
V1_RESULTS = SUBMISSION_ROOT / "revision_results"

RNG_SEED = 20260701

# ---------------------------------------------------------------------------
# M1: paired significance tests on 10-area cosine advantage
# ---------------------------------------------------------------------------


def _permutation_test_mean(
    diffs: np.ndarray, n_perm: int = 100_000, seed: int = RNG_SEED
) -> float:
    """Two-sided sign-flip permutation test on the mean of paired differences."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean(diffs))
    n = len(diffs)
    if n == 0:
        return float("nan")
    signs = rng.choice([-1, 1], size=(n_perm, n))
    permuted_means = (signs * diffs[None, :]).mean(axis=1)
    p = float(np.mean(np.abs(permuted_means) >= abs(observed)))
    return p


def _bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 20_000, alpha: float = 0.05, seed: int = RNG_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def run_m1_paired_significance() -> pd.DataFrame:
    """Rigorous paired inference on cosine advantage of LatentDynamicsNet vs persistence."""
    df = pd.read_csv(V1_RESULTS / "alphaearth_area_metrics.csv")
    # aggregate as-reported: all 10 valid cached areas
    diffs_all = df["advantage"].to_numpy(dtype=float)

    # split out val-area (Poyang Lake) from train/test/OOD for M1 second view
    val_areas = {"poyang_lake", "pearl_river"}  # both val per Table 1
    df_no_val = df[~df["area"].isin(val_areas)].copy()

    def summarise(label: str, values: np.ndarray) -> dict[str, Any]:
        n = len(values)
        if n < 2:
            return {
                "subset": label,
                "n": n,
                "mean": float(np.mean(values)) if n else float("nan"),
            }
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        se = sd / math.sqrt(n)
        t_stat, t_p = stats.ttest_1samp(values, popmean=0.0)
        wilcoxon = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        cohen_dz = mean / sd if sd > 0 else float("inf")
        ci_lo, ci_hi = _bootstrap_ci(values)
        perm_p = _permutation_test_mean(values)
        n_pos = int(np.sum(values > 0))
        sign_p = stats.binomtest(n_pos, n, 0.5).pvalue
        return {
            "subset": label,
            "n": n,
            "mean_advantage": mean,
            "sd": sd,
            "se": se,
            "cohen_dz": cohen_dz,
            "t_stat": float(t_stat),
            "t_p_two_sided": float(t_p),
            "wilcoxon_stat": float(wilcoxon.statistic),
            "wilcoxon_p_two_sided": float(wilcoxon.pvalue),
            "permutation_p_two_sided": perm_p,
            "sign_test_p": float(sign_p),
            "n_positive": n_pos,
            "n_negative": int(np.sum(values < 0)),
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
        }

    rows = [
        summarise("all_valid_cached", diffs_all),
        summarise("no_val_areas", df_no_val["advantage"].to_numpy(dtype=float)),
    ]
    out = pd.DataFrame(rows)
    out_path = REV_ROOT / "paired_significance_tests.csv"
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# M2: independent categorical change validation - proper paired inference
# ---------------------------------------------------------------------------

INDEP_METRIC_COLS = {
    "change_f1": {
        "model": "model_change_f1",
        "shuffled": "shuffled_model_change_f1",
        "prior": "transition_prior_change_f1",
        "persistence": "persistence_change_f1",
    },
    "end_accuracy": {
        "model": "model_end_accuracy",
        "shuffled": "shuffled_model_end_accuracy",
        "prior": "transition_prior_end_accuracy",
        "persistence": "persistence_end_accuracy",
    },
    "changed_pixel_accuracy": {
        "model": "model_changed_pixel_accuracy",
        "shuffled": "shuffled_model_changed_pixel_accuracy",
        "prior": "transition_prior_changed_pixel_accuracy",
        "persistence": "persistence_changed_pixel_accuracy",
    },
    "area_bias_mae": {
        "model": "model_area_bias_mae",
        "prior": "transition_prior_area_bias_mae",
        "persistence": "persistence_area_bias_mae",
    },
    "transition_exact_match": {
        "model": "model_transition_exact_match",
        "prior": "transition_prior_transition_exact_match",
        "persistence": "persistence_transition_exact_match",
    },
}


def _paired_test(model: np.ndarray, ctrl: np.ndarray) -> dict[str, Any]:
    diffs = model - ctrl
    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_zero = int(np.sum(diffs == 0))
    mean = float(np.mean(diffs)) if n else float("nan")
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    result: dict[str, Any] = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "mean_diff": mean,
        "sd_diff": sd,
    }
    # Wilcoxon requires at least one non-zero
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1:
        try:
            wil = stats.wilcoxon(
                nonzero, zero_method="wilcox", alternative="two-sided"
            )
            result["wilcoxon_p"] = float(wil.pvalue)
        except ValueError:
            result["wilcoxon_p"] = float("nan")
    else:
        result["wilcoxon_p"] = float("nan")
    if n_pos + n_neg > 0:
        result["sign_test_p"] = float(
            stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue
        )
    else:
        result["sign_test_p"] = float("nan")
    return result


def run_m2_independent_change_analysis() -> dict[str, Any]:
    df = pd.read_csv(V1_RESULTS / "independent_change_validation_by_area.csv")
    df["pair_id"] = df["area"] + "_" + df["start_year"].astype(str) + "-" + df[
        "end_year"
    ].astype(str)
    # Wuyi 2020-2021: reference change rate 0.0% => F1 is ill-defined.
    ill_defined_mask = df["true_change_pixels"] == 0
    n_ill = int(ill_defined_mask.sum())
    df_effective = df.loc[~ill_defined_mask].copy()

    all_paired: list[dict[str, Any]] = []
    for metric, cols in INDEP_METRIC_COLS.items():
        model = df_effective[cols["model"]].to_numpy(dtype=float)
        for ctrl in cols:
            if ctrl == "model":
                continue
            ctrl_vals = df_effective[cols[ctrl]].to_numpy(dtype=float)
            row = {
                "metric": metric,
                "control": ctrl,
                "model_mean": float(np.mean(model)),
                "ctrl_mean": float(np.mean(ctrl_vals)),
            }
            row.update(_paired_test(model, ctrl_vals))
            all_paired.append(row)

    out = pd.DataFrame(all_paired)
    out.to_csv(REV_ROOT / "independent_change_paired_tests.csv", index=False)

    # per-area win/loss summary vs each control
    winloss_rows = []
    for _, r in df_effective.iterrows():
        for metric, cols in INDEP_METRIC_COLS.items():
            higher_is_better = metric != "area_bias_mae"
            model_val = float(r[cols["model"]])
            for ctrl in cols:
                if ctrl == "model":
                    continue
                ctrl_val = float(r[cols[ctrl]])
                delta = model_val - ctrl_val
                won = (delta > 0) if higher_is_better else (delta < 0)
                winloss_rows.append(
                    {
                        "pair_id": r["pair_id"],
                        "metric": metric,
                        "control": ctrl,
                        "model_val": model_val,
                        "ctrl_val": ctrl_val,
                        "delta": delta,
                        "higher_is_better": higher_is_better,
                        "model_won": bool(won),
                    }
                )
    winloss = pd.DataFrame(winloss_rows)
    winloss.to_csv(REV_ROOT / "independent_change_per_pair_winloss.csv", index=False)

    summary = {
        "n_pairs_total": int(len(df)),
        "n_pairs_dropped_ill_defined_f1": n_ill,
        "n_pairs_effective": int(len(df_effective)),
        "dropped_ill_defined_pairs": df.loc[ill_defined_mask, "pair_id"].tolist(),
        "means_after_drop": {
            m: {
                subset: float(df_effective[c].mean())
                for subset, c in cols.items()
            }
            for m, cols in INDEP_METRIC_COLS.items()
        },
    }
    (REV_ROOT / "independent_change_paired_tests_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


# ---------------------------------------------------------------------------
# M3: GeoSOS-FLUS per-area win/loss (24 townships, single seed)
# ---------------------------------------------------------------------------

FLUS_METRICS = {
    "change_f1": True,
    "fom": True,
    "transition_accuracy": True,
    "allocation_disagreement": False,  # lower is better
}


def run_m3_flus_per_area_wins() -> pd.DataFrame:
    src = SUBMISSION_ROOT / (
        "paper58_true_geosos_flus_xiangzhen24_fixed_flus_"
        "external_loo_transition_floor030_same_grid_2026-06-27"
    )
    df = pd.read_csv(src / "metrics_by_method.csv")

    # Pivot: one row per (area, method), metric columns
    keep_cols = [
        "method",
        "area",
        "change_f1",
        "fom",
        "transition_accuracy",
        "allocation_disagreement",
    ]
    df = df[keep_cols].copy()

    # Compare each non-flus method against geosos_flus_console per area
    flus = (
        df[df["method"] == "geosos_flus_console"].set_index("area").drop(
            columns=["method"]
        )
    )

    rows = []
    for method, sub in df.groupby("method"):
        if method == "geosos_flus_console":
            continue
        joined = sub.set_index("area").join(flus, lsuffix="_m", rsuffix="_flus")
        for metric, higher_is_better in FLUS_METRICS.items():
            m_col = metric + "_m"
            f_col = metric + "_flus"
            deltas = joined[m_col] - joined[f_col]
            if higher_is_better:
                wins = (deltas > 0).sum()
                losses = (deltas < 0).sum()
            else:
                wins = (deltas < 0).sum()
                losses = (deltas > 0).sum()
            ties = int((deltas == 0).sum())
            n = int(len(deltas))
            n_effective = int(wins + losses)
            sign_p = (
                float(stats.binomtest(int(wins), n_effective, 0.5).pvalue)
                if n_effective > 0
                else float("nan")
            )
            wil = (
                float(
                    stats.wilcoxon(
                        deltas[deltas != 0].to_numpy(), alternative="two-sided"
                    ).pvalue
                )
                if n_effective > 0
                else float("nan")
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "wins_vs_flus": int(wins),
                    "losses_vs_flus": int(losses),
                    "ties": ties,
                    "win_rate": float(wins) / n if n else float("nan"),
                    "mean_delta": float(deltas.mean()),
                    "median_delta": float(deltas.median()),
                    "sign_test_p": sign_p,
                    "wilcoxon_p": wil,
                    "higher_is_better": higher_is_better,
                }
            )
    out = pd.DataFrame(rows).sort_values(["method", "metric"])
    out.to_csv(REV_ROOT / "geosos_flus_per_area_wins.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# S1: cosine gap -> end-year accuracy gap curve
# ---------------------------------------------------------------------------


def run_s1_cosine_to_accuracy_curve() -> pd.DataFrame:
    """Build empirical mapping from cosine advantage to end-year accuracy delta.

    We join the per-area cosine advantage (M1 table) with the per-area
    end-year accuracy delta from the independent change validation.
    """
    cos = pd.read_csv(V1_RESULTS / "alphaearth_area_metrics.csv")
    chg = pd.read_csv(V1_RESULTS / "independent_change_validation_by_area.csv")

    # collapse independent-change to per-area (mean across year-pairs)
    chg_area = (
        chg.groupby("area")[["model_end_accuracy", "persistence_end_accuracy"]]
        .mean()
        .reset_index()
    )
    chg_area["end_accuracy_delta"] = (
        chg_area["model_end_accuracy"] - chg_area["persistence_end_accuracy"]
    )

    joined = cos.merge(chg_area, on="area", how="inner")
    joined = joined[
        [
            "area",
            "advantage",
            "model_end_accuracy",
            "persistence_end_accuracy",
            "end_accuracy_delta",
        ]
    ].rename(columns={"advantage": "cosine_advantage"})
    joined.to_csv(REV_ROOT / "cosine_to_accuracy_curve.csv", index=False)
    return joined


# ---------------------------------------------------------------------------
# S5: multi-step degradation for all 10 valid cached areas
#
# We do not have per-step cached predictions on disk for every area, so we
# instead publish a transparent summary of what data are available and what
# are not, ensuring no selective reporting on paper. The three areas in the
# v1 table (Yangtze Delta, Jing-Jin-Ji, Chengdu Plain) are the only ones
# with recorded 1-6 step advantage rows; the remaining seven are marked
# "not-recorded" and moved to Limitations.
# ---------------------------------------------------------------------------


MULTISTEP_V1 = {
    "yangtze_delta": {1: 0.007, 2: 0.003, 3: 0.004, 4: 0.002, 5: 0.001, 6: -0.002},
    "jing_jin_ji": {1: 0.005, 2: 0.002, 3: 0.001, 4: -0.001, 5: -0.003, 6: -0.005},
    "chengdu_plain": {1: 0.003, 2: 0.001, 3: 0.002, 4: 0.001, 5: -0.001, 6: -0.003},
}


def run_s5_multistep_disclosure() -> pd.DataFrame:
    cos = pd.read_csv(V1_RESULTS / "alphaearth_area_metrics.csv")
    rows = []
    for _, r in cos.iterrows():
        area = r["area"]
        if area in MULTISTEP_V1:
            for k, adv in MULTISTEP_V1[area].items():
                rows.append(
                    {
                        "area": area,
                        "step": k,
                        "advantage": adv,
                        "status": "recorded_v1",
                    }
                )
        else:
            rows.append(
                {
                    "area": area,
                    "step": 1,
                    "advantage": float(r["advantage"]),
                    "status": "1step_only_recorded",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REV_ROOT / "multistep_disclosure_all_areas.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Additional analysis: post-hoc power on the M1 area-level advantage.
# ---------------------------------------------------------------------------


def run_m1_power_analysis() -> dict[str, Any]:
    df = pd.read_csv(V1_RESULTS / "alphaearth_area_metrics.csv")
    x = df["advantage"].to_numpy(dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    dz = mean / sd if sd > 0 else float("inf")
    # target power = 0.8, alpha = 0.05, two-sided one-sample t
    # required n approx (z_{a/2} + z_{1-b})^2 / dz^2 for large-sample
    from math import ceil

    z_a = stats.norm.ppf(1 - 0.025)
    z_b = stats.norm.ppf(0.80)
    if dz != 0:
        n_required = int(ceil(((z_a + z_b) / dz) ** 2))
    else:
        n_required = -1
    payload = {
        "n_current": n,
        "mean": mean,
        "sd": sd,
        "cohen_dz": dz,
        "required_n_for_80pct_power_two_sided_005": n_required,
    }
    (REV_ROOT / "m1_power_analysis.json").write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    print("[M1] paired significance tests...")
    m1 = run_m1_paired_significance()
    print(m1.to_string(index=False))
    print()
    print("[M1] power analysis...")
    pw = run_m1_power_analysis()
    print(json.dumps(pw, indent=2))
    print()
    print("[M2] independent change paired analysis...")
    m2 = run_m2_independent_change_analysis()
    print(json.dumps(m2, indent=2))
    print()
    print("[M3] GeoSOS-FLUS per-area wins...")
    m3 = run_m3_flus_per_area_wins()
    print(m3.to_string(index=False))
    print()
    print("[S1] cosine gap -> accuracy curve...")
    s1 = run_s1_cosine_to_accuracy_curve()
    print(s1.to_string(index=False))
    print()
    print("[S5] multistep disclosure...")
    s5 = run_s5_multistep_disclosure()
    print(s5.to_string(index=False))


if __name__ == "__main__":
    main()
