#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

audio_dir=$1

# Comma-separated list of audio files to ignore
# (detail: https://www.notion.so/IEICE-System-paper-5aa3a3edb1804b62adedc3bd0dbdc5fc?pvs=4#95f10b9a2bb8477ea4d0ee86d6661f62)
ignore_files="626_pred.wav,1240_pred.wav,101_pred.wav,683_pred.wav,634_pred.wav,1669_pred.wav,1812_pred.wav,1299_pred.wav,2631_pred.wav,1676_pred.wav,1657_pred.wav,319_pred.wav,2518_pred.wav,1069_pred.wav,2518_pred.wav,2357_pred.wav,461_pred.wav,682_pred.wav"

python scripts/score_utmos.py \
  --audio_dir $audio_dir \
  --ignore_files $ignore_files | tee `dirname $audio_dir`/score_utmos.txt
