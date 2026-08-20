#!/usr/bin/env bash

# =============================================================================
# LM-head evaluation for the SPECIAL-TOKEN experiment

set -euo pipefail

SEED=42
PROBING_ROOT="probing"

CJK_DICT="${PROBING_ROOT}/synset_pos_artificial_cjk.json"
HIRA_DICT="${PROBING_ROOT}/synset_pos_artificial_hiragana.json"
EVAL_SCRIPT="${PROBING_ROOT}/lm_head_eval_special.py"

RUN_DIR="${PROBING_ROOT}/st_probing_run"

FINAL_OMITTED="${RUN_DIR}/final_omitted.json"
PARALLEL="${RUN_DIR}/final_omitted_corpus/parallel_corpus_synset.json"

for arm in shared none disjoint; do

    MODEL_DIR="${RUN_DIR}/treatment_${arm}_seed${SEED}/final"
    OUT_DIR="${PROBING_ROOT}/lmhead_st_${arm}_seed${SEED}"

    echo
    echo "============================================================"
    echo "LM-head evaluation: special-token ${arm}"
    echo "============================================================"
    echo "Model:           ${MODEL_DIR}"
    echo "Final omitted:   ${FINAL_OMITTED}"
    echo "Parallel corpus: ${PARALLEL}"
    echo "Output:          ${OUT_DIR}"
    echo

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: treatment model directory does not exist:"
        echo "  ${MODEL_DIR}"
        exit 1
    fi

    if [[ ! -f "${MODEL_DIR}/special_config.json" ]]; then
        echo "ERROR: special_config.json does not exist:"
        echo "  ${MODEL_DIR}/special_config.json"
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
echo "  probing/lmhead_st_shared_seed${SEED}/"
echo "  probing/lmhead_st_none_seed${SEED}/"
echo "  probing/lmhead_st_disjoint_seed${SEED}/"
echo "============================================================"
