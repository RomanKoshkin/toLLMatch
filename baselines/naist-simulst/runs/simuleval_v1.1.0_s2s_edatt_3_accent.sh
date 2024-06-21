#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# Experimantal settings
tgt_l=$1    # de, zh or ja
EXP_NAME=$2
ckpt_name=$3 # checkpoint_best.pt
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name

# SimulEval settings
who=tst-COMMON # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
source_list=${evaldir}/${who}.wav_list
target=${evaldir}/${who}.${tgt_l}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
if [ $tgt_l == "de" ]; then
  echo "German TTS is not supported."
  exit 1
  sacrebleu_tokenizer=13a
elif [ $tgt_l == "ja" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2s_edatt_3_accent.py
  sacrebleu_tokenizer=ja-mecab
elif [ $tgt_l == "zh" ]; then
  echo "Chinese TTS is not supported."
  exit 1
fi
chunk_size=$4 # in ms, default=250
attn_threshold=$5
frame_num=$6
exclude_last_attn=
beam=$7  # default=1
tag=$8  # si, off

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=speech_to_speech/${ckpt}.${who}.${tgt_l}.alpha${attn_threshold}-lambda${frame_num}-chunk${chunk_size}-beam${beam}_3_accent
results=${SAVE_DIR}/${EXP_NAME}/simuleval_v1.1.0/${simuleval_log_dir_name}
if [ -n "$tag" ]; then
  results=${results}-tag_${tag}
fi

mkdir -p $results && cp ${BASH_SOURCE[0]} $results
#TMP touch ${results}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")
cmd="simuleval
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
--sacrebleu-tokenizer $sacrebleu_tokenizer
--quality-metrics WHISPER_ASR_BLEU
--latency-metrics StartOffset EndOffset ATD
--target-speech-lang ja
--start-index 1
--end-index 30"
if [ -n "$tag" ]; then
  cmd=""${cmd}"
  --tag $tag"
fi
eval $cmd

sed 's/\t/,/g' ${results}/scores.tsv > ${results}/scores.csv

# get timestamps
python scripts/simulst/get_timestamp.py \
  ${MUSTC_v2_ROOT_AHC}/en-${tgt_l}/data/${who}/wav \
  ${MUSTC_v2_ROOT_AHC}/en-${tgt_l}/data/${who}/txt/${who}.yaml \
  ${results}/instances.log
