#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

set -eu

src=en
export LC_ALL=C
export ASR_DATADIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/asr/${src}; mkdir -p ${ASR_DATADIR}
SPM_MODEL=/ahc/work/sst-team/IWSLT2023/data/training/en-de/spm_bpe250000_st
KALDI_ROOT=/project/nakamura-lab09/Work/yuka-ko/espnet/tools/kaldi
sph2pipe=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe


function preprocess_voxpopuli(){
    echo "###########################"
    echo "starting voxpopuli preprocessing"
    echo "###########################"
    export VOXPOPULI_ROOT_AHC=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/voxpopuli/data_asr/transcribed_data/en
    export VOXPOPULI_DATADIR=${ASR_DATADIR}/voxpopuli_asr/data; mkdir -p ${VOXPOPULI_DATADIR}
    export LOGDIR=${VOXPOPULI_DATADIR}/log; mkdir -p ${LOGDIR}
    cp ${BASH_SOURCE[0]} ${VOXPOPULI_DATADIR}

    # link
    # ln -s /ahc/data/VoxPopuli/voxpopuli/data_asr/raw_audios /ahc/work3/sst-team/IWSLT2023/work/yuka-ko/data/dataset/voxpopuli/data_asr/raw_audios
    # transcript, segment
    # python -m voxpopuli.get_asr_data --root ${VOXPOPULI_ROOT_AHC} --lang en

    python ${UPC_IWSLT_ROOT}/scripts/data_prep_asr/prep_voxpopuli_asr_data.py \
        --data-root ${VOXPOPULI_ROOT_AHC} --task asr --use-audio-input \
        --only-manifest \
        --out-root ${VOXPOPULI_DATADIR}/processed \
        --vocab-type bpe --vocab-size 250000 | tee -a ${LOGDIR}/preprocess.log
}

function main(){
    preprocess_voxpopuli
}

main

