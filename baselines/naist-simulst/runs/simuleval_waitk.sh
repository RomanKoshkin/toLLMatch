# Usage: simuleval_waitk.sh $who $chunk_size $waitk
# Example: bash runs/simuleval_waitk.sh tst-COMMON 250 3

#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# Experimantal settings
EXP_NAME=add_st_corpora_de_lna_hubert_2.5e-4
ckpt_name=checkpoint_best.pt
ckpt_file=${SAVE_DIR}/${EXP_NAME}/ckpts/$ckpt_name
tgt_l=de    # de, zh or ja

# SimulEval settings
who=$1 # [tst-COMMON, toy, toy.1]
evaldir=${ROOT}/data/eval_data/en-${tgt_l}/evaldata
source_list=${evaldir}/${who}.wav_list
target=${evaldir}/${who}.${tgt_l}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
agent=${SCRIPT_ROOT}/simulst/agents/waitk.py
chunk_size=$2 # in ms, default=250
waitk=$3  # default=3

# Output settings
ckpt=`echo $ckpt_file | sed -r 's/.*_(.*)\..*/\1/'`
simuleval_log_dir_name=${ckpt}.${who}.wait${waitk}-chunk${chunk_size}-beam1
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
--waitk $waitk
--port $port"
cmd=""${cmd}" 2>&1 | tee -a ${results}/simuleval.log"
eval $cmd

python ${SCRIPT_ROOT}/simulst/log2gen.py ${results}/instances.log
