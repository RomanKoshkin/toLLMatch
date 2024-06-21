#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

tgt_l=$1
hyp_file=$2

who=tst-COMMON # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
ref_file=${evaldir}/${who}.${tgt_l}

bleurt_model=${ROOT}/models/BLEURT-20

python scripts/scores.py \
  $ref_file $hyp_file $bleurt_model | tee `dirname $hyp_file`/score_bleurt.txt
