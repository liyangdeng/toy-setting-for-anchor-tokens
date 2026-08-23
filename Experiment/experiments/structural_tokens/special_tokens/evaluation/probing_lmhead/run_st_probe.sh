#!/usr/bin/env bash

# =============================================================================
# Linear probe for the SPECIAL-TOKEN experiment

set -euo pipefail

SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"

MLP_ROOT="$REPO_ROOT/Experiment/evaluation/masked_language_probing"
BUILD_ROOT="$MLP_ROOT/build_probing_corpus"

ST_ROOT="$REPO_ROOT/Experiment/experiments/structural_tokens/special_tokens"
CORPUS_ROOT="$ST_ROOT/corpora"

PROBING_ROOT="$SCRIPT_DIR/probing_runs"
RESULT_ROOT="$SCRIPT_DIR/probe_results"
RUN_DIR="${PROBING_ROOT}/st_probing_run"

BUILD_PROBING="$BUILD_ROOT/build_probing_corpus.py"
DEPRIVED="$BUILD_ROOT/deprived_triples.json"
OMITTED="$BUILD_ROOT/omitted_triples.json"
PROBE_MANIFEST="$BUILD_ROOT/probe_manifest.json"

GEN_SCRIPT="$REPO_ROOT/data/generate_sentences/v3_generate_sentences.py"
BUILD_CORPUS="$REPO_ROOT/data/corpus/build_synset_corpus.py"
GRAMMAR="$REPO_ROOT/data/grammar/grammar_templates_adj.py"

CJK_DICT="$CORPUS_ROOT/synset_pos_artificial_cjk.json"
HIRA_DICT="$CORPUS_ROOT/synset_pos_artificial_hiragana.json"

FINAL_TRAIN_ST="$(find "$ST_ROOT" -type f -name 'final_train_st.py' -print -quit 2>/dev/null || true)"
LINEAR_PROBE_ST="$(find "$REPO_ROOT" -type f -name 'linear_probe_special_final.py' -print -quit 2>/dev/null || true)"
MONO_TRAIN="$(find "$REPO_ROOT" -type f -name 'train_monolingual_synset.py' -print -quit 2>/dev/null || true)"

for required in \
    "$BUILD_PROBING" \
    "$DEPRIVED" \
    "$OMITTED" \
    "$PROBE_MANIFEST" \
    "$GEN_SCRIPT" \
    "$BUILD_CORPUS" \
    "$GRAMMAR" \
    "$CJK_DICT" \
    "$HIRA_DICT"
do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing required file: $required"
        exit 1
    fi
done

if [[ -z "$FINAL_TRAIN_ST" || ! -f "$FINAL_TRAIN_ST" ]]; then
    echo "ERROR: could not find final_train_st.py under $ST_ROOT"
    exit 1
fi

if [[ -z "$LINEAR_PROBE_ST" || ! -f "$LINEAR_PROBE_ST" ]]; then
    echo "ERROR: could not find linear_probe_special_final.py under $REPO_ROOT"
    exit 1
fi

if [[ -z "$MONO_TRAIN" || ! -f "$MONO_TRAIN" ]]; then
    echo "ERROR: could not find train_monolingual_synset.py under $REPO_ROOT"
    exit 1
fi

mkdir -p "$PROBING_ROOT" "$RESULT_ROOT"

echo "============================================================"
echo "BUILD PROBING CORPUS"
echo "============================================================"

python "$BUILD_PROBING" \
    --deprived_triples "$DEPRIVED" \
    --omitted_triples  "$OMITTED" \
    --probe_manifest   "$PROBE_MANIFEST" \
    --gen_script          "$GEN_SCRIPT" \
    --build_corpus_script "$BUILD_CORPUS" \
    --mono_train_script   "$MONO_TRAIN" \
    --grammar             "$GRAMMAR" \
    --cjk_dict  "$CJK_DICT" \
    --hira_dict "$HIRA_DICT" \
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

    python "$FINAL_TRAIN_ST" \
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
    OUT_DIR="${RESULT_ROOT}/res_probe_st_${arm}_seed${SEED}"

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

    python "$LINEAR_PROBE_ST" \
        --model_dir "${MODEL_DIR}" \
        --final_omitted "${FINAL_OMITTED}" \
        --parallel "${PARALLEL}" \
        --cjk_dict "$CJK_DICT" \
        --hira_dict "$HIRA_DICT" \
        --seed "${SEED}" \
        --out_dir "${OUT_DIR}"

done



echo
echo "Treatment models:"
echo "  ${RUN_DIR}/treatment_shared_seed${SEED}/final"
echo "  ${RUN_DIR}/treatment_none_seed${SEED}/final"
echo "  ${RUN_DIR}/treatment_disjoint_seed${SEED}/final"
echo
echo "Probe results:"
echo "  ${RESULT_ROOT}/res_probe_st_shared_seed${SEED}/"
echo "  ${RESULT_ROOT}/res_probe_st_none_seed${SEED}/"
echo "  ${RESULT_ROOT}/res_probe_st_disjoint_seed${SEED}/"
