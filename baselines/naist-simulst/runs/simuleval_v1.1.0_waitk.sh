# Usage: simuleval_waitk.sh $who $chunk_size $waitk
# Example: bash runs/simuleval_waitk.sh tst-COMMON 250 3

#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# Experimantal settings
tgt_l=$1    # de, zh or ja
EXP_NAME=$2
# add_st_corpora_de_lna_hubert_2.5e-4, ja_lna_hubert_2.5e-4_ablation_add-covost
ckpt_name=$3 # checkpoint_best.pt
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name

# SimulEval settings
who=tst-COMMON # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
source_list=${evaldir}/${who}.wav_list
target=${evaldir}/${who}.${tgt_l}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
if [ $tgt_l == "de" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_waitk_word.py
  sacrebleu_tokenizer=13a
elif [ $tgt_l == "ja" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_waitk_char.py
  sacrebleu_tokenizer=ja-mecab
elif [ $tgt_l == "zh" ]; then
  agent=${SCRIPT_ROOT}/simulst/agents/v1.1.0/s2t_waitk_char.py
  sacrebleu_tokenizer=zh
fi
chunk_size=$4 # in ms, default=250
waitk=$5  # default=3
tag=$6  # si, off

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=${ckpt}.${who}.${tgt_l}.wait${waitk}-chunk${chunk_size}-beam1
results=${SAVE_DIR}/${EXP_NAME}/simuleval_v1.1.0/${simuleval_log_dir_name}
if [ -n "$tag" ]; then
  results=${results}-tag_${tag}
fi

mkdir -p $results && cp ${BASH_SOURCE[0]} $results
touch ${results}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")
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
--waitk $waitk
--remote-port $port
--gpu
--sacrebleu-tokenizer $sacrebleu_tokenizer
--computation-aware"
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
