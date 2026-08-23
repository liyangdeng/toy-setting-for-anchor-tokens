#!/usr/bin/env bash

# =============================================================================
# Linear probe for the SPECIAL-TOKEN experiment

set -euo pipefail

SEED=42
PROBING_ROOT="probing"
RUN_DIR="${PROBING_ROOT}/st_probing_run"

echo "============================================================"
echo "BUILD PROBING CORPUS"
echo "============================================================"

python probing/build_probing_corpus.py \
    --deprived_triples probing/deprived_triples.json \
    --omitted_triples  probing/omitted_triples.json \
    --probe_manifest   probing/probe_manifest.json \
    --gen_script          probing/v3_generate_sentences.py \
    --build_corpus_script probing/build_synset_corpus.py \
    --mono_train_script   probing/train_monolingual_synset.py \
    --grammar             probing/grammar_templates_adj.py \
    --cjk_dict  probing/synset_pos_artificial_cjk.json \
    --hira_dict probing/synset_pos_artificial_hiragana.json \
    --seed "${SEED}" \
    --out_dir "${RUN_DIR}"


echo "============================================================"
echo "TRAIN OMITTED/TREATMENT MODELS"
echo "============================================================"

CORPUS_A="${RUN_DIR}/a_training.txt"
CORPUS_B="${RUN_DIR}/b_training.txt"

if [[ ! -f "${CORPUS_A}" ]]; then
    echo "ERROR: missing ${CORPUS_A}"
    exit 1
fi

if [[ ! -f "${CORPUS_B}" ]]; then
    echo "ERROR: missing ${CORPUS_B}"
    exit 1
fi

for arm in shared none disjoint; do

    MODEL_DIR="${RUN_DIR}/treatment_${arm}_seed${SEED}"

    echo
    echo "=== Training ST treatment model: ${arm} ==="
    echo "A corpus: ${CORPUS_A}"
    echo "B corpus: ${CORPUS_B}"
    echo "Output:   ${MODEL_DIR}"

    python final_train_st.py \
        --setting "${arm}" \
        --seed "${SEED}" \
        --corpus_a "${CORPUS_A}" \
        --corpus_b "${CORPUS_B}" \
        --output_dir "${MODEL_DIR}"

done

echo
echo "============================================================"
echo "LINEAR PROBE"
echo "============================================================"

FINAL_OMITTED="${RUN_DIR}/final_omitted.json"
PARALLEL="${RUN_DIR}/final_omitted_corpus/parallel_corpus_synset.json"

for arm in shared none disjoint; do

    MODEL_DIR="${RUN_DIR}/treatment_${arm}_seed${SEED}/final"
    OUT_DIR="${PROBING_ROOT}/probe_st_${arm}_seed${SEED}"

    echo
    echo "=== Linear probe: special-token ${arm} ==="
    echo "Model:           ${MODEL_DIR}"
    echo "Final omitted:   ${FINAL_OMITTED}"
    echo "Parallel corpus: ${PARALLEL}"
    echo "Output:          ${OUT_DIR}"

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: treatment model not found:"
        echo "  ${MODEL_DIR}"
        exit 1
    fi

    if [[ ! -f "${MODEL_DIR}/special_config.json" ]]; then
        echo "ERROR: special_config.json not found:"
        echo "  ${MODEL_DIR}/special_config.json"
        exit 1
    fi

    if [[ ! -f "${FINAL_OMITTED}" ]]; then
        echo "ERROR: final_omitted.json not found:"
        echo "  ${FINAL_OMITTED}"
        exit 1
    fi

    if [[ ! -f "${PARALLEL}" ]]; then
        echo "ERROR: parallel corpus not found:"
        echo "  ${PARALLEL}"
        exit 1
    fi

    python probing/linear_probe_special_final.py \
        --model_dir "${MODEL_DIR}" \
        --final_omitted "${FINAL_OMITTED}" \
        --parallel "${PARALLEL}" \
        --cjk_dict probing/synset_pos_artificial_cjk.json \
        --hira_dict probing/synset_pos_artificial_hiragana.json \
        --seed "${SEED}" \
        --out_dir "${OUT_DIR}"

done


echo
echo "============================================================"
echo "Done"
echo "============================================================"
echo
echo "Treatment models:"
echo "  ${RUN_DIR}/treatment_shared_seed${SEED}/final"
echo "  ${RUN_DIR}/treatment_none_seed${SEED}/final"
echo "  ${RUN_DIR}/treatment_disjoint_seed${SEED}/final"
echo
echo "Probe results:"
echo "  probing/probe_st_shared_seed${SEED}/"
echo "  probing/probe_st_none_seed${SEED}/"
echo "  probing/probe_st_disjoint_seed${SEED}/"
