
# checking the data length for mt
for tgt in de ja zh; do
    DIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/en-${tgt}/mt/v1/
    logdir=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/en-${tgt}/mt/v1/log; mkdir -p ${logdir}
    log=${logdir}/check-data-len.log
    for name in $(ls ${DIR}); do
        echo ${DIR}/${name} | tee -a ${log}
        cat ${DIR}/${name} | tee -a ${log}
    done
done
