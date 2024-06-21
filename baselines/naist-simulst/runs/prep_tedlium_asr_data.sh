#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

set -eu

src=en
export LC_ALL=C
export ASR_DATADIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/asr/${src}; mkdir -p ${ASR_DATADIR}
cp ${BASH_SOURCE[0]} ${ASR_DATADIR}/
SPM_MODEL=/ahc/work/sst-team/IWSLT2023/data/training/en-de/spm_bpe250000_st
KALDI_ROOT=/project/nakamura-lab09/Work/yuka-ko/espnet/tools/kaldi
sph2pipe=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe
function preprocess_tedlium(){
    echo "###########################"
    echo "starting TEDLIUM v3 preprocessing"
    echo "###########################"
    # doc:
    # The official test set is not part of this corpus, 
    # but if you want to use the development data 
    # you need to make sure that it is not part of the data
    # root:
    export TEDLIUM_ROOT_AHC=/ahc/data/TED-LIUM/v3/TEDLIUM_release-3/legacy
    export TEDLIUM_DATADIR=${ASR_DATADIR}/tedlium_v3/data

    # convert sph2wav in full size
    for split in dev test train; do
        export TEDLIUM_SPLIT_TXT=${TEDLIUM_DATADIR}/${split}/txt; mkdir -p ${TEDLIUM_SPLIT_TXT}
        export TEDLIUM_SPLIT_WAV=${TEDLIUM_DATADIR}/${split}/wav; mkdir -p ${TEDLIUM_SPLIT_WAV}

        stmdir=${TEDLIUM_ROOT_AHC}/${split}/stm/
        sphdir=${TEDLIUM_ROOT_AHC}/${split}/sph/

        for sphname in $(ls ${sphdir}); do
            sphpath=${sphdir}/${sphname}
            wavpath=${TEDLIUM_SPLIT_WAV}/${sphname%.sph}.wav
            if [ ! -e ${wavpath} ]; then
                sph2pipe -f wav ${sphpath} > ${wavpath}
            fi
        done
        echo "wav making done"
    done

    python ${UPC_IWSLT_ROOT}/scripts/data_prep_asr/prep_tedlium_asr_data.py \
        --data-root ${TEDLIUM_DATADIR} --task asr --use-audio-input \
        --only-manifest \
        --out-root ${TEDLIUM_DATADIR}/processed \
        --vocab-type bpe --vocab-size 250000 \
        --make-tsv \
        --origin-data-root ${TEDLIUM_ROOT_AHC}
}

function main(){
    preprocess_tedlium
}

main

