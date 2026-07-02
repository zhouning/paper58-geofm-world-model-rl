#!/usr/bin/env bash
# Paper 58 RSE R2 — macOS experiment orchestrator (E1 through E7).
# Abort-safe, resumable, and prints an aggregate summary at the end.
set -uo pipefail  # no -e so a single experiment failure does not kill the batch

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---------- argument parsing ----------
MODE="all"          # all | only | resume
ONLY=""
SMOKE_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)      MODE="all";    shift ;;
        --resume)   MODE="resume"; shift ;;
        --only)     MODE="only";   ONLY="${2:-}"; shift 2 ;;
        --smoke)    SMOKE_FLAG="--smoke"; shift ;;
        -h|--help)
            echo "Usage: bash run_all_macos.sh [--all|--resume|--only e1,e2,...] [--smoke]"
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- experiment registry ----------
# Keep this compatible with macOS /bin/bash 3.2; associative arrays require bash 4.
experiment_cmd() {
    case "$1" in
        e1_extract)       echo "e1_prithvi_patch_extract.py --all" ;;
        e1_train)         echo "e1_prithvi_patch_train_eval.py --seeds 42 123 456 --epochs 50" ;;
        e2_terrain)       echo "e2_terrain_ablation.py --seeds 42 123 456 --epochs 50" ;;
        e3_multistep)     echo "e3_multistep_all_areas.py" ;;
        e4_decoder)       echo "e4_per_year_decoder.py" ;;
        e5_sensitivity)   echo "e5_sa_alloc_sensitivity.py" ;;
        e6_expand)        echo "e6_expand_areas.py" ;;
        e7_third_encoder) echo "e7_third_encoder.py" ;;
        *)                return 1 ;;
    esac
}

# canonical execution order
EXP_ORDER=(e1_extract e1_train e2_terrain e3_multistep e4_decoder e5_sensitivity e6_expand e7_third_encoder)

# ---------- filter based on mode ----------
declare -a SELECTED
if [[ "$MODE" == "only" ]]; then
    IFS=',' read -ra ONLY_ARR <<< "$ONLY"
    for prefix in "${ONLY_ARR[@]}"; do
        for exp in "${EXP_ORDER[@]}"; do
            if [[ "$exp" == "$prefix"* ]]; then
                SELECTED+=("$exp")
            fi
        done
    done
else
    SELECTED=("${EXP_ORDER[@]}")
fi

# ---------- runner ----------
log_root="$HERE/logs"
mkdir -p "$log_root"
started=$(date +%s)
run_summary_json="$log_root/run_$(date +%Y%m%d_%H%M%S).json"
SELECTED_COUNT=${#SELECTED[@]}

echo "===== Paper 58 R2 macOS orchestrator ====="
echo "started at $(date)"
echo "mode: $MODE  smoke: ${SMOKE_FLAG:-none}"
if [[ $SELECTED_COUNT -gt 0 ]]; then
    echo "selected: ${SELECTED[*]}"
else
    echo "selected:"
fi
echo ""

RESULT_STATUS_KEYS=()
RESULT_STATUS_VALS=()
RESULT_WALL_VALS=()

record_result() {
    RESULT_STATUS_KEYS+=("$1")
    RESULT_STATUS_VALS+=("$2")
    RESULT_WALL_VALS+=("$3")
}

lookup_result() {
    key="$1"
    field="$2"
    i=0
    while [[ $i -lt ${#RESULT_STATUS_KEYS[@]} ]]; do
        if [[ "${RESULT_STATUS_KEYS[$i]}" == "$key" ]]; then
            if [[ "$field" == "status" ]]; then
                echo "${RESULT_STATUS_VALS[$i]}"
            else
                echo "${RESULT_WALL_VALS[$i]}"
            fi
            return 0
        fi
        i=$((i + 1))
    done
    if [[ "$field" == "status" ]]; then
        echo "UNKNOWN"
    else
        echo "0"
    fi
}

if [[ $SELECTED_COUNT -gt 0 ]]; then
for exp in "${SELECTED[@]}"; do
    cmd_tail="$(experiment_cmd "$exp")"
    script_name="${cmd_tail%% *}"
    args="${cmd_tail#* }"
    if [[ "$args" == "$script_name" ]]; then
        args=""
    fi
    logfile="$log_root/${exp}.log"

    # Resume support: skip only full-run sentinels. Smoke runs write
    # .done_smoke and must not block later full --resume execution.
    exp_prefix="${exp%%_*}"   # e1, e2, ...
    results_dir="$HERE/results/${exp_prefix}_"*
    if [[ "$MODE" == "resume" ]]; then
        skip=false
        for rd in $results_dir; do
            [[ -d "$rd" && -e "$rd/.done_full" ]] && skip=true
            [[ -d "$rd" && -e "$rd/.done" && ! -e "$rd/.done_smoke" ]] && skip=true
        done
        if $skip; then
            echo "[$exp] SKIP (already full .done); logfile=$logfile"
            record_result "$exp" "SKIPPED" "0"
            continue
        fi
    fi

    echo "---- [$exp] running python $script_name${args:+ $args}${SMOKE_FLAG:+ $SMOKE_FLAG} ----"
    t0=$(date +%s)
    python "$HERE/$script_name" $args $SMOKE_FLAG > "$logfile" 2>&1
    rc=$?
    t1=$(date +%s)
    wall=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        record_result "$exp" "OK" "$wall"
        echo "[$exp] OK (${wall}s)  log: $logfile"
    elif [[ $rc -eq 2 ]]; then
        record_result "$exp" "SKIPPED_MISSING_DEP" "$wall"
        echo "[$exp] SKIPPED (rc=2, missing external dependency); log: $logfile"
    else
        record_result "$exp" "FAILED_rc=$rc" "$wall"
        echo "[$exp] FAILED (rc=$rc); see $logfile"
        # print the last 20 lines of the failing log
        echo "  last 20 lines:"
        tail -20 "$logfile" | sed 's/^/    /'
    fi
done
fi

ended=$(date +%s)
elapsed=$((ended - started))

# ---------- summary JSON ----------
{
    echo "{"
    echo "  \"started\": $started,"
    echo "  \"ended\": $ended,"
    echo "  \"wall_s\": $elapsed,"
    echo "  \"mode\": \"$MODE\","
    echo "  \"smoke\": \"$SMOKE_FLAG\","
    echo "  \"experiments\": {"
    first=1
    if [[ $SELECTED_COUNT -gt 0 ]]; then
    for exp in "${SELECTED[@]}"; do
        if [[ $first -eq 0 ]]; then echo ","; fi
        status="$(lookup_result "$exp" "status")"
        wall="$(lookup_result "$exp" "wall")"
        printf '    "%s": {"status": "%s", "wall_s": %s}' \
            "$exp" "$status" "$wall"
        first=0
    done
    fi
    echo ""
    echo "  }"
    echo "}"
} > "$run_summary_json"

echo ""
echo "===== summary ====="
echo "total wall clock: ${elapsed}s ($((elapsed/60)) min)"
echo "run summary JSON: $run_summary_json"
echo ""
if [[ $SELECTED_COUNT -gt 0 ]]; then
for exp in "${SELECTED[@]}"; do
    status="$(lookup_result "$exp" "status")"
    wall="$(lookup_result "$exp" "wall")"
    printf "  %-24s %-10s %ss\n" "$exp" "$status" "$wall"
done
fi
echo ""
echo "next steps after everything is OK:"
echo "  git add experiments/macos_r2/results/ experiments/macos_r2/weights/"
echo "  git commit -m \"Paper 58 R2 macOS-side experiments complete\""
echo "  git push origin main"
