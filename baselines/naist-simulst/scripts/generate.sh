#!/bin/bash

EXP_NAME=$1
CKPT_NAME=$2
SUBSET=$3
TGT_LANG=$4

ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$CKPT_NAME

if [ "$TGT_LANG" == "de" ]; then
  tokenizer=13a
elif [ "$TGT_LANG" == "ja" ]; then
  tokenizer=ja-mecab
elif [ "$TGT_LANG" == "zh" ]; then
  tokenizer=zh
fi

fairseq-generate ${DATA_ROOT}/en-${TGT_LANG}/ \
--path ${ckpt_file} \
--task speech_to_text \
--gen-subset $SUBSET \
--seed 48151623 \
--prefix-size 1 \
--scoring sacrebleu \
--max-source-positions 900_000 \
--max-target-positions 1024 \
--max-tokens 1_920_000 \
--beam 5 \
--sacrebleu-tokenizer $tokenizer
