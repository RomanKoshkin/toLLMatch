#!/bin/bash

# change the target language in `--data-file` to de/es/it/fr/ru and `--tgt-lang` to `deu/spa/ita/fra/rus`

ABS_PATH_TO_PROJECT_ROOT=<ABS_PATH_TO_PROJECT_ROOT> # specify abs path to toLLMmatch
END_INDEX=2
streaming_evaluate \
    --task s2tt \
    --data-file $ABS_PATH_TO_PROJECT_ROOT/baselines/seamless-streaming/ted_tst_2024_en_de.tsv \
    --audio-root-dir $ABS_PATH_TO_PROJECT_ROOT/raw_datasets/ted-tst-2024 \
    --output $ABS_PATH_TO_PROJECT_ROOT/baselines/seamless-streaming/out \
    --tgt-lang deu \
    --source-segment-size 400 \
    --decision-threshold 0.9 \
    --start-index 0 \
    --end-index $END_INDEX