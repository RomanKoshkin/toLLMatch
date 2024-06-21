#!/bin/bash

. $(dirname $0)/path.sh || exit 1;

export LIBRI_ROOT_AHC=/ahc/data/LibriSpeech/LibriSpeech/
src=en
export TEDLIUM_ROOT_AHC=/ahc/data/TED-LIUM/v3
export ASR_DATADIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/asr/${src}; mkdir -p ${ASR_DATADIR}

SPM_MODEL=/ahc/work/sst-team/IWSLT2023/data/training/en-de/spm_bpe250000_st

export LIBRI_DATADIR=${ASR_DATADIR}/librispeech
cp ${BASH_SOURCE[0]} ${LIBRI_DATADIR}

python ${UPC_IWSLT_ROOT}/scripts/data_prep_asr/prep_librispeech_asr_data.py \
    --data-root ${LIBRI_ROOT_AHC} --task asr --use-audio-input \
    --only-manifest --append-lang-id --out-root ${LIBRI_DATADIR} \
    --vocab-type bpe --vocab-size 250000

