# Usage: simuleval_holdn.sh $device $who $chunk_size $holdn $beam
# Example: bash runs/simuleval_hold-n.sh 0 tst-COMMON 500 6 5

#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# GPU setting
device=$1  # set "-1 to run on CPU"
export CUDA_VISIBLE_DEVICES=${device}

# Experimantal settings
EXP_NAME=add_st_corpora_de_lna_hubert_2.5e-4
ckpt_name=checkpoint_best.pt
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name
tgt_l=de    # de, zh or ja

# SimulEval settings
who=$2 # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
source_list=${evaldir}/${who}.wav_list
target=${evaldir}/${who}.${tgt_l}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
agent=${SCRIPT_ROOT}/simulst/agents/holdn.py
chunk_size=$3 # in ms, default=500
holdn=$4  # default=6
beam=$5   # default=1 

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=${ckpt}.${who}.hold${holdn}-chunk${chunk_size}-beam${beam}
results=${SAVE_DIR}/${EXP_NAME}/simuleval/${simuleval_log_dir_name}

mkdir $results && cp ${BASH_SOURCE[0]} $results
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
--chunk-size $chunk_size
--holdn $holdn
--port $port
--beam $beam"
if [ $device -ne -1 ]; then
  cmd=${cmd}" --gpu"
fi
cmd=""${cmd}" 2>&1 | tee -a ${results}/simuleval.log"
eval $cmd

python ${SCRIPT_ROOT}/simulst/log2gen.py ${results}/instances.log
