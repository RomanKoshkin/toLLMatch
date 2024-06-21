#!/bin/bash
sleep 3
OUTPUT_DIR=results/ende
PATH_TO_PROJECT_ROOT=<ABS_PATH_TO_PROJECT_ROOT>
simuleval \
    --agent $PATH_TO_PROJECT_ROOT/baselines/FBK-fairseq/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_edatt.py \
    --source $PATH_TO_PROJECT_ROOT/evaluation/SOURCES/ted_tst_2024_102 \
    --target $PATH_TO_PROJECT_ROOT/evaluation/OFFLINE_TARGETS/ted_tst_2024_de_102.txt \
    --data-bin $PATH_TO_PROJECT_ROOT/baselines/FBK-fairseq/DATA_ROOT \
    --config $PATH_TO_PROJECT_ROOT/baselines/FBK-fairseq/DATA_ROOT/config_simul.yaml \
    --model-path $PATH_TO_PROJECT_ROOT/baselines/FBK-fairseq/DATA_ROOT/checkpoint_avg7.pt \
    --extract-attn-from-layer 3 \
    --frame-num 2 \
    --attn-threshold 0.3 \
    --speech-segment-factor 20 \
    --output $PATH_TO_PROJECT_ROOT/baselines/FBK-fairseq/results \
    --port 3333 \
    --gpu \
    --scores \
    --end-index 2