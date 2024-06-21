#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

expname=prelim1
TOOL_ROOT=./tools
src=en

for tgt in en ja zh; do
    LANG=(${src} ${tgt})
    langpair=${src}-${tgt}
    SPM_PREF=/ahc/work/sst-team/IWSLT2023/data/training/${langpair}/spm_bpe250000_st
    RAW_ORIG_DIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/${langpair}/mt/v1
    RAW_DIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/${langpair}/mt/v2; mkdir -p ${RAW_DIR}

    EXP_DIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/${langpair}/mt/${expname}; mkdir -p ${EXP_DIR}
    PRO_DIR=${EXP_DIR}/processed; mkdir -p ${PRO_DIR}
    BIN_DIR=${EXP_DIR}/data-bin; mkdir -p ${BIN_DIR}
    echo ${EXP_DIR}
    cp ${BASH_SOURCE[0]} ${EXP_DIR}

    ###########################
    # small change from v1 to v2 (rstrip)
    ###########################
    # [FIXME] only use first time
    # for filename in $(ls ${RAW_ORIG_DIR}/); do
    #     cat ${RAW_ORIG_DIR}/${filename} | python3 -c "import sys; [print(l.replace('\u2028',' ').rstrip('\r\n')) for l in sys.stdin]" > ${RAW_DIR}/${filename}
    # done
    ###########################
    # select necessary data
    ###########################
    # [INFO] data length
    # log: /ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/en-de/mt/v1/log/check-data-len.log
    # script: /ahc/work/sst-team/IWSLT2023/work/yuka-ko/iwslt-2023/utils/check-data-len.sh
    # [INFO] data head (see written language)
    # log: /ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/en-de/mt/v1/log/check-data-head.log
    # script: /ahc/work/sst-team/IWSLT2023/work/yuka-ko/iwslt-2023/utils/check-data-head.sh

    if [ ${tgt} = de ]; then
        TRN_PREF=(${RAW_DIR}/train.newscom.v16 ${RAW_DIR}/train.opus2018 ${RAW_DIR}/train.ted ${RAW_DIR}/train.wikimatrix ${RAW_DIR}/train.wikititles ${RAW_DIR}/train.paracrawl.v9 ${RAW_DIR}/train.rapid2019 ${RAW_DIR}/train-commoncrawl)
    elif [ ${tgt} = ja ]; then
        TRN_PREF=(${RAW_DIR}/train.newscom.v16 ${RAW_DIR}/train.opus2018 ${RAW_DIR}/train.ted ${RAW_DIR}/train.wikimatrix ${RAW_DIR}/train.wikititles ${RAW_DIR}/train.jparacrawl.v3 ${RAW_DIR}/train.kftt ${RAW_DIR}/train.jesc)
    elif [ ${tgt} = zh ]; then
        TRN_PREF=(${RAW_DIR}/train.newscom.v16 ${RAW_DIR}/train.opus2018 ${RAW_DIR}/train.ted ${RAW_DIR}/train.wikimatrix ${RAW_DIR}/train.wikititles ${RAW_DIR}/train.paracrawl.v9)
    fi
    # [INFO] dev and test following en-de (not en-ja) https://ahcweb01.naist.jp/papers/conference/2022/202205_IWSLT_yasumasa-k/202205_IWSLT_yasumasa-k.paper.pdf
    DEV_PREF=(${RAW_DIR}/dev.dev2010 ${RAW_DIR}/dev.tst2010 ${RAW_DIR}/dev.tst2011 ${RAW_DIR}/dev.tst2012)
    # [FIXME] add dev2021
    TST_PREF=(${RAW_DIR}/dev.tst2015)

    # [BUGFIX] *_O_PREF of en-de not to be accumulated when en-ja
    unset TRN_O_PREF DEV_O_PREF TST_O_PREF
    for trn in "${TRN_PREF[@]##*/}"; do TRN_O_PREF+=(${PRO_DIR}/${trn}); done
    ############################
    # sentencepiece encoding
    ############################
    ALL_PREF=("${TRN_PREF[@]}" "${DEV_PREF[@]}" "${TST_PREF[@]}")
    for pref in "${ALL_PREF[@]}"
    do
        for lang in "${LANG[@]}"
        do
            python ${TOOL_ROOT}/src/spm_encode.py \
                --input ${pref}.${lang} \
                --output ${PRO_DIR}/${pref##*/}.${lang} \
                --model ${SPM_PREF}.model || exit 1;

            echo ${pref}.${lang}
            cat ${PRO_DIR}/${pref##*/}.${lang} | wc -l
        done
    done
    ####################
    # gathering train data
    ####################
    rm ${PRO_DIR}/train.all.*
    for trn in ${TRN_O_PREF[@]}; do
        cat ${trn}.${src} >> ${PRO_DIR}/train.all.${src}
        cat ${trn}.${tgt} >> ${PRO_DIR}/train.all.${tgt}
    done
    echo ${PRO_DIR}/train.all.${src} $(cat ${PRO_DIR}/train.all.${src} | wc -l)
    echo ${PRO_DIR}/train.all.${tgt} $(cat ${PRO_DIR}/train.all.${tgt} | wc -l)

    ##########################
    # fairseq-preprocess
    ##########################
    for val in "${DEV_PREF[@]##*/}"; do DEV_O_PREF+=(${PRO_DIR}/${val}); done
    for tst in "${TST_PREF[@]##*/}"; do TST_O_PREF+=(${PRO_DIR}/${tst}); done
    echo ${TRN_O_PREF[@]} | sed 's/ /,/g'
    echo ${DEV_O_PREF[@]} | sed 's/ /,/g'
    echo ${TST_O_PREF[@]} | sed 's/ /,/g'
    rm -rf ${BIN_DIR}
    cmd="fairseq-preprocess
    --source-lang ${LANG[0]}
    --target-lang ${LANG[1]}
    --trainpref ${PRO_DIR}/train.all
    --validpref `echo ${DEV_O_PREF[@]} | sed 's/ /,/g'`
    --testpref `echo ${TST_O_PREF[@]} | sed 's/ /,/g'`
    --destdir $BIN_DIR
    --joined-dictionary
    --srcdict ${SPM_PREF}.txt
    --workers 1"
    eval $cmd
done
