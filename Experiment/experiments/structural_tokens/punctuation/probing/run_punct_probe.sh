#!/usr/bin/env bash

# =============================================================================
# Linear probe for the PUNCTUATION experiment

set -euo pipefail

SEED=42
PROBING_ROOT="probing"


echo "============================================================"
echo "BUILD PROBING CORPORA"
echo "============================================================"


# ---------- SHARED ----------

echo
echo "=== Building probing corpus: shared ==="

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
    --out_dir probing/punct_probing_shared


# ---------- NONE ----------

echo
echo "=== Building probing corpus: none ==="

python probing/build_probing_corpus.py \
    --deprived_triples probing/deprived_triples.json \
    --omitted_triples  probing/omitted_triples.json \
    --probe_manifest   probing/probe_manifest.json \
    --gen_script          probing/v3_generate_sentences.py \
    --build_corpus_script probing/punct_to_lp_bridge_nopunct.py \
    --mono_train_script   probing/train_monolingual_synset.py \
    --grammar             probing/grammar_templates_adj.py \
    --cjk_dict  probing/synset_pos_artificial_cjk.json \
    --hira_dict probing/synset_pos_artificial_hiragana.json \
    --seed "${SEED}" \
    --out_dir probing/punct_probing_none


# ---------- DISJOINT ----------

echo
echo "=== Building probing corpus: disjoint ==="

python probing/build_probing_corpus.py \
    --deprived_triples probing/deprived_triples.json \
    --omitted_triples  probing/omitted_triples.json \
    --probe_manifest   probing/probe_manifest.json \
    --gen_script          probing/v3_generate_sentences.py \
    --build_corpus_script probing/punct_to_lp_bridge_disjoint.py \
    --mono_train_script   probing/train_monolingual_synset.py \
    --grammar             probing/grammar_templates_adj.py \
    --cjk_dict  probing/synset_pos_artificial_cjk.json \
    --hira_dict probing/synset_pos_artificial_hiragana.json \
    --seed "${SEED}" \
    --out_dir probing/punct_probing_disjoint


# =============================================================================
# STAGE 2: TRAIN OMITTED/TREATMENT MODELS
# =============================================================================

echo
echo "============================================================"
echo "Stage 2: training omitted/treatment models"
echo "============================================================"

for arm in shared none disjoint; do

    RUN_DIR="${PROBING_ROOT}/punct_probing_${arm}"
    MODEL_DIR="${RUN_DIR}/treatment_seed${SEED}"

    CORPUS_A="${RUN_DIR}/a_training.txt"
    CORPUS_B="${RUN_DIR}/b_training.txt"

    echo
    echo "=== Training treatment model: ${arm} ==="
    echo "A corpus: ${CORPUS_A}"
    echo "B corpus: ${CORPUS_B}"
    echo "Output:   ${MODEL_DIR}"

    # Sanity checks
    if [[ ! -f "${CORPUS_A}" ]]; then
        echo "ERROR: missing ${CORPUS_A}"
        exit 1
    fi

    if [[ ! -f "${CORPUS_B}" ]]; then
        echo "ERROR: missing ${CORPUS_B}"
        exit 1
    fi

    python final_train_punct.py \
        --setting "${arm}" \
        --seed "${SEED}" \
        --corpus_a "${CORPUS_A}" \
        --corpus_b "${CORPUS_B}" \
        --output_dir "${MODEL_DIR}"

done


# =============================================================================
# STAGE 3: LINEAR PROBE
# =============================================================================

echo
echo "============================================================"
echo "Stage 3: running linear probes"
echo "============================================================"

for arm in shared none disjoint; do

    RUN_DIR="${PROBING_ROOT}/punct_probing_${arm}"
    MODEL_DIR="${RUN_DIR}/treatment_seed${SEED}/final"
    OUT_DIR="${PROBING_ROOT}/probe_punct_${arm}_seed${SEED}"

    FINAL_OMITTED="${RUN_DIR}/final_omitted.json"
    PARALLEL="${RUN_DIR}/final_omitted_corpus/parallel_corpus_synset.json"

    echo
    echo "=== Linear probe: ${arm} ==="
    echo "Model:           ${MODEL_DIR}"
    echo "Final omitted:   ${FINAL_OMITTED}"
    echo "Parallel corpus: ${PARALLEL}"
    echo "Output:          ${OUT_DIR}"

    # Sanity checks
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: treatment model not found:"
        echo "  ${MODEL_DIR}"
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

    python probing/linear_probe.py \
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
echo "  probing/punct_probing_shared/treatment_seed${SEED}/final"
echo "  probing/punct_probing_none/treatment_seed${SEED}/final"
echo "  probing/punct_probing_disjoint/treatment_seed${SEED}/final"
echo
echo "Probe results:"
echo "  probing/probe_punct_shared_seed${SEED}/"
echo "  probing/probe_punct_none_seed${SEED}/"
echo "  probing/probe_punct_disjoint_seed${SEED}/"
