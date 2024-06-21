#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

set -eu

src=en
export LC_ALL=C
export ASR_DATADIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/asr/${src}; mkdir -p ${ASR_DATADIR}
SPM_MODEL=/ahc/work/sst-team/IWSLT2023/data/training/en-de/spm_bpe250000_st
KALDI_ROOT=/project/nakamura-lab09/Work/yuka-ko/espnet/tools/kaldi
sph2pipe=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe

function preprocess_commonvoice_v6.1(){
    echo "###########################"
    echo "starting common voice v6.1 preprocessing"
    echo "###########################"
    export COMMONVOICE_ROOT_AHC=/ahc/data/Common_Voice_Corpus/6.1/cv-corpus-6.1-2020-12-11/en/
    export COMMONVOICE_DATADIR=${ASR_DATADIR}/commonvoice/6.1/data; mkdir -p ${COMMONVOICE_DATADIR}
    cp ${BASH_SOURCE[0]} ${COMMONVOICE_DATADIR}
    export LOGDIR=${COMMONVOICE_DATADIR}/log; mkdir -p ${LOGDIR}

    mkdir -p ${COMMONVOICE_DATADIR}/wav
    
    # # mp3 to wav
    # python utils/mp3_to_wav.py \
    #     --mp3-dir ${COMMONVOICE_ROOT_AHC}/clips \
    #     --wav-dir ${COMMONVOICE_DATADIR}/wav \
    #     --split-dir ./split_clip_list/

    python ${UPC_IWSLT_ROOT}/scripts/data_prep_asr/prep_commonvoice_asr_data.py \
        --data-root ${COMMONVOICE_ROOT_AHC} --task asr \
        --out-root ${COMMONVOICE_DATADIR} --use-audio-input \
        --only-manifest | tee -a ${LOGDIR}/preprocess.v2.log
}

function main(){
    preprocess_commonvoice_v6.1
}

main
