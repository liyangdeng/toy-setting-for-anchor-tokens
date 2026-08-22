#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=/opt/miniconda3/bin/python3
REPO=/Users/pengyuwen/toy-setting-for-anchor-tokens
CJK_DICT=$REPO/data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_cjk_edges_adj_augmented.json
HIRA_DICT=$REPO/data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_hiragana_edges_adj_augmented.json
CONFIGS="000 001 011 111"

echo "===== STAGE: mono (4 in parallel) ====="
for cfg in $CONFIGS; do
  $PY build_probing_corpus_necessity.py --stage mono --cfg $cfg > cfg_$cfg/mono_log.txt 2>&1 &
done
wait
echo "===== mono done ====="

echo "===== STAGE: filter (sequential, fast) ====="
for cfg in $CONFIGS; do
  $PY build_probing_corpus_necessity.py --stage filter --cfg $cfg > cfg_$cfg/filter_log.txt 2>&1
  echo "  filter $cfg done"
done

echo "===== STAGE: assemble (sequential, fast) ====="
for cfg in $CONFIGS; do
  $PY build_probing_corpus_necessity.py --stage assemble --cfg $cfg > cfg_$cfg/assemble_log.txt 2>&1
  echo "  assemble $cfg done"
done

echo "===== STAGE: multi retrain (4 in parallel) ====="
for cfg in $CONFIGS; do
  $PY $REPO/Experiment/training/train_multilingual_synset.py \
    --corpus_a cfg_$cfg/a_training.txt \
    --corpus_b cfg_$cfg/b_training.txt \
    --output_dir cfg_$cfg/multi_probe \
    --seed 42 > cfg_$cfg/multi_train_log.txt 2>&1 &
done
wait
echo "===== multi retrain done ====="

echo "===== STAGE: linear_probe.py (entity track) ====="
for cfg in $CONFIGS; do
  $PY $REPO/Experiment/evaluation/masked_language_probing/probing/linear_probe.py \
    --track entity \
    --model_dir cfg_$cfg/multi_probe/final \
    --final_omitted cfg_$cfg/final_omitted.json \
    --parallel cfg_$cfg/final_omitted_parallel.json \
    --cjk_dict $CJK_DICT \
    --hira_dict $HIRA_DICT \
    --out_dir cfg_$cfg/probe_results > cfg_$cfg/probe_log.txt 2>&1
  echo "  probe $cfg done"
done

echo "===== ALL STAGES COMPLETE ====="
