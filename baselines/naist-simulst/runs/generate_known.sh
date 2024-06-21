#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=$1

# Experimantal settings
EXP_NAME=$2 # e.g. de_lna_hubert_2.5e-4_ablation_no-covost
who=$3 # e.g. tst-COMMON_mustc
#DEBUG: EXP_NAME=add_st_corpora_de_lna_hubert_2.5e-4
#DEBUG: who=toy.1
ckpt_name=checkpoint_best.pt
ckpt_name=checkpoint_ave5.pt
tgt_l=ja    # de, zh or ja

# Decoding settings
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name
beam=5
prefix_size=${4:-1}  # set 4 if use tagging otherwise 1

if [ "$tgt_l" == "de" ]; then
  tokenizer=13a
elif [ "$tgt_l" == "ja" ]; then
  tokenizer=ja-mecab
elif [ "$tgt_l" == "zh" ]; then
  tokenizer=zh
fi

ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
results=${SAVE_DIR}/${EXP_NAME}/gen.${ckpt}.${who}.beam${beam}.prefix${prefix_size}
mkdir -p $results && cp ${BASH_SOURCE[0]} $results
touch ${results}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")
cmd="fairseq-generate ${DATA_ROOT}/en-${tgt_l}
--path $ckpt_file
--task speech_to_text
--gen-subset $who
--seed 48151623
--prefix-size $prefix_size
--scoring sacrebleu
--max-source-positions 900_000
--max-target-positions 1024
--max-tokens 1_920_000
--beam $beam
--sacrebleu-tokenizer $tokenizer
--results-path $results"
cmd=""${cmd}" 2>&1 | tee -a ${results}/generate.log"
echo $cmd
eval $cmd

cd $results
grep -e ^T generate-${who}.txt | sed s/^T-//g | python ${SCRIPT_ROOT}/rerank.py > ${who}.T
grep -e ^D generate-${who}.txt | sed s/^D-//g | cut -f1,3 | python ${SCRIPT_ROOT}/rerank.py> ${who}.D
sed -i "s/^<off>//" "${who}.T"
sed -i "s/^<si>//" "${who}.T"
sed -i "s/^<off>//" "${who}.D"
sed -i "s/^<si>//" "${who}.D"
cat ${who}.D | sacrebleu ${who}.T -tok $tokenizer | tee ${who}.sacrebleu
