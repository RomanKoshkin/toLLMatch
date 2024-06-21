#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# Experimantal settings
tgt_l=$1    # de, zh or ja
EXP_NAME=$2  # e.g. EXP_NAME=prefix_alignment_fixed_st-langid-interconnection_filtered_max4000_lr2.5e-5 
ckpt_name=checkpoint_best.pt
beam=5  # default=1
chunk_size=$3 # in ms, default=250
attn_threshold=$4
frame_num=$5
exclude_last_attn=
tag=  # si, off
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name

# SimulEval settings
who=tst-COMMON # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
source_list=${evaldir}/${who}.wav_list
target=${evaldir}/${who}.${tgt_l}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
if [ $tgt_l == "de" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_edatt_word.py
  sacrebleu_tokenizer=13a
elif [ $tgt_l == "ja" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_edatt_char.py
  sacrebleu_tokenizer=ja-mecab
elif [ $tgt_l == "zh" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_edatt_char.py
  sacrebleu_tokenizer=zh
fi

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=${ckpt}.${who}.${tgt_l}.alpha${attn_threshold}-lambda${frame_num}-chunk${chunk_size}-beam${beam}
results=${SAVE_DIR}/${EXP_NAME}/simuleval_v1.1.0/debug_edatt_nca/${simuleval_log_dir_name}
if [ -n "$tag" ]; then
  results=${results}-tag_${tag}
fi

mkdir -p $results && cp ${BASH_SOURCE[0]} $results
touch ${results}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")
cmd="time simuleval
--agent $agent
--source $source_list
--target $target
--model-path $ckpt_file
--data-bin ${DATA_ROOT}/en-${tgt_l}
--use-audio-input
--output $results
--lang $tgt_l
--source-segment-size $chunk_size
--beam $beam
--attn-threshold $attn_threshold
--frame-num $frame_num
--remote-port $port
--gpu
--sacrebleu-tokenizer $sacrebleu_tokenizer"
#--computation-aware  # calculate computation-aware latency
#--end-index 10  # run on 10 samples
if [ $tgt_l == "de" ]; then
  cmd=""${cmd}"
  --sentencepiece-model ${MBART_ROOT}/sentence.bpe.model"
elif [ $tgt_l == "ja" ]; then
  cmd=""${cmd}"
  --eval-latency-unit char
  --filtered-tokens '▁'"
elif [ $tgt_l == "zh" ]; then
  cmd=""${cmd}"
  --eval-latency-unit char
  --filtered-tokens '▁'"
fi
if [ -n "$tag" ]; then
  cmd=""${cmd}"
  --tag $tag"
fi
cmd=""${cmd}" 2>&1 | tee -a ${results}/simuleval.log"
eval $cmd

python ${SCRIPT_ROOT}/simulst/log2gen.py ${results}/instances.log
cat ${results}/generation.txt | sacrebleu $target -tok $sacrebleu_tokenizer | tee ${results}/${who}.sacrebleu

sed 's/\t/,/g' ${results}/scores.tsv > ${results}/scores.csv

# get timestamps
python scripts/simulst/get_timestamp.py \
  ${MUSTC_v2_ROOT_AHC}/en-${tgt_l}/data/${who}/wav \
  ${MUSTC_v2_ROOT_AHC}/en-${tgt_l}/data/${who}/txt/${who}.yaml \
  ${results}/instances.log
