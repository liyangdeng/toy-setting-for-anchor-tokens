#!/usr/bin/env bash

# =============================================================================
# LM-head evaluation for the PUNCTUATION experiment

set -euo pipefail

SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"

PUNCT_ROOT="$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation"
CORPUS_ROOT="$PUNCT_ROOT/corpora"
LMHEAD_ROOT="$REPO_ROOT/Experiment/evaluation/lm_head"

PROBING_ROOT="$SCRIPT_DIR/probing_runs"
RESULT_ROOT="$SCRIPT_DIR/lm_head_results"

CJK_DICT="$CORPUS_ROOT/synset_pos_artificial_cjk.json"
HIRA_DICT="$CORPUS_ROOT/synset_pos_artificial_hiragana.json"
EVAL_SCRIPT="$LMHEAD_ROOT/lm_head_eval.py"

mkdir -p "$RESULT_ROOT"

for arm in shared none disjoint; do

    RUN_DIR="${PROBING_ROOT}/punct_probing_${arm}"

    # IMPORTANT: use the bilingual omitted/treatment model
    MODEL_DIR="${RUN_DIR}/treatment_seed${SEED}/final"

    FINAL_OMITTED="${RUN_DIR}/final_omitted.json"
    PARALLEL="${RUN_DIR}/final_omitted_corpus/parallel_corpus_synset.json"

    OUT_DIR="${RESULT_ROOT}/lmhead_punct_${arm}_seed${SEED}"

    echo
    echo "============================================================"
    echo "LM-head evaluation: punctuation ${arm}"
    echo "============================================================"
    echo "Model:           ${MODEL_DIR}"
    echo "Final omitted:   ${FINAL_OMITTED}"
    echo "Parallel corpus: ${PARALLEL}"
    echo "Output:          ${OUT_DIR}"
    echo

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: model directory does not exist:"
        echo "  ${MODEL_DIR}"
        exit 1
    fi

    if [[ ! -f "${FINAL_OMITTED}" ]]; then
        echo "ERROR: final_omitted.json does not exist:"
        echo "  ${FINAL_OMITTED}"
        exit 1
    fi

    if [[ ! -f "${PARALLEL}" ]]; then
        echo "ERROR: parallel corpus does not exist:"
        echo "  ${PARALLEL}"
        exit 1
    fi

    if [[ ! -f "${CJK_DICT}" ]]; then
        echo "ERROR: CJK dictionary does not exist:"
        echo "  ${CJK_DICT}"
        exit 1
    fi

    if [[ ! -f "${HIRA_DICT}" ]]; then
        echo "ERROR: Hiragana dictionary does not exist:"
        echo "  ${HIRA_DICT}"
        exit 1
    fi

    python "${EVAL_SCRIPT}" \
        --model_dir "${MODEL_DIR}" \
        --final_omitted "${FINAL_OMITTED}" \
        --parallel "${PARALLEL}" \
        --cjk_dict "${CJK_DICT}" \
        --hira_dict "${HIRA_DICT}" \
        --out_dir "${OUT_DIR}"

done

echo
echo "Results:"
echo "  ${RESULT_ROOT}/lmhead_punct_shared_seed${SEED}/"
echo "  ${RESULT_ROOT}/lmhead_punct_none_seed${SEED}/"
echo "  ${RESULT_ROOT}/lmhead_punct_disjoint_seed${SEED}/"