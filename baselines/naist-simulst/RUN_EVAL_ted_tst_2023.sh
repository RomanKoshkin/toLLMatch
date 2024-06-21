#!/bin/bash

ABS_PATH_TO_PROJECT_ROOT= #<ABS_PATH_TO_PROJECT_ROOT> # specify abs path to toLLMmatch
ABS_PATH_TO_DATA_BIN=
ABS_PATH_TO_MODEL_CHECKPOINT=
END_INDEX=2
OUTPUT_DIR=results/ende
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2t_la_word.py \
  --sentencepiece-model $ABS_PATH_TO_DATA_BIN/spm_bpe250000_st.model \
  --model-path $ABS_PATH_TO_MODEL_CHECKPOINT/checkpoint_best.pt \
  --data-bin $ABS_PATH_TO_DATA_BIN \
  --source $ABS_PATH_TO_PROJECT_ROOT/evaluation/SOURCES/src_ted_new_tst_100_abspath.de \
  --target $ABS_PATH_TO_PROJECT_ROOT/evaluation/OFFLINE_TARGETS/tgt_ted_new_tst_100.de \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang de \
  --source-segment-size 950 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer 13a \
  --end-index $END_INDEX