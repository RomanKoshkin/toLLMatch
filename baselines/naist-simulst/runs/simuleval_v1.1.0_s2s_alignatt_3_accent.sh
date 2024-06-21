#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

ts=`date +"%Y%m%d%I%M%S"`

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
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2s_alignatt_3_accent.py
  sacrebleu_tokenizer=ja-mecab
elif [ $tgt_l == "zh" ]; then
  echo "Chinese TTS is not supported."
  exit 1
fi
chunk_size=$4 # in ms, default=250
frame_num=$5
beam=$6  # default=1
tag=$7  # si, off

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=speech_to_speech/${ts}.${ckpt}.${who}.${tgt_l}.alignatt.chunk${chunk_size}-beam${beam}-frame${frame_num}_3_accent
results=${SAVE_DIR}/${EXP_NAME}/simuleval_v1.1.0/${simuleval_log_dir_name}
if [ -n "$tag" ]; then
  results=${results}-tag_${tag}
fi

mkdir -p $results && cp ${BASH_SOURCE[0]} $results
#TMP touch ${results}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")

# TTS models paths
SUB2YOMI_PATH=${TTS_MODELS_PATH}/base_model3/sub2yomi/output0.out
YOMI2TTS_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/checkpoint_64000.pth.tar
TTS2WAV_PATH=${TTS_MODELS_PATH}/base_model3/tts2wav/checkpoint_400000.pth.tar
SUB2YOMI_DICT_PATH=${TTS_MODELS_PATH}/base_model3/sub2yomi/vocabs_thd1.dict
YOMI2TTS_P_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/phoneme.json
YOMI2TTS_A1_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a1.json
YOMI2TTS_A2_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a2.json
YOMI2TTS_A3_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a3.json
YOMI2TTS_F1_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/f1.json
YOMI2TTS_F2_DICT_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/f2.json

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
--frame-num $frame_num
--exclude-first-attn
--remote-port $port
--gpu
--sacrebleu-tokenizer $sacrebleu_tokenizer
--quality-metrics WHISPER_ASR_BLEU
--latency-metrics StartOffset EndOffset ATD
--target-speech-lang ja
--sub2yomi_model_path $SUB2YOMI_PATH
--yomi2tts_model_path $YOMI2TTS_PATH
--tts2wav_model_path $TTS2WAV_PATH
--sub2yomi_dict_path $SUB2YOMI_DICT_PATH
--yomi2tts_phoneme_dict_path $YOMI2TTS_P_DICT_PATH
--yomi2tts_a1_dict_path $YOMI2TTS_A1_DICT_PATH
--yomi2tts_a2_dict_path $YOMI2TTS_A2_DICT_PATH
--yomi2tts_a3_dict_path $YOMI2TTS_A3_DICT_PATH
--yomi2tts_f1_dict_path $YOMI2TTS_F1_DICT_PATH
--yomi2tts_f2_dict_path $YOMI2TTS_F2_DICT_PATH"
#--start-index 0
#--end-index 10"
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
