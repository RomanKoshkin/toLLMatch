#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=$1

# (1) Data Preparation
# MuST-C (en-de,en-zh,en-ja)
for task in {asr,st}; do
    python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_mustc_data.py \
    --data-root $MUSTC_v2_ROOT_AHC --task $task --use-audio-input \
    --only-manifest --append-lang-id --out-root $MUSTC_v2_ROOT \
    --vocab-type bpe --vocab-size 250000
done
cp ${MBART_ROOT}/dict.en_XX.txt ${MUSTC_v2_ROOT}/en-de/spm_bpe250000_asr.txt
cp ${MBART_ROOT}/dict.de_DE.txt ${MUSTC_v2_ROOT}/en-de/spm_bpe250000_st.txt
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-de/spm_bpe250000_asr.model
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-de/spm_bpe250000_st.model
cp ${MBART_ROOT}/dict.en_XX.txt ${MUSTC_v2_ROOT}/en-zh/spm_bpe250000_asr.txt
cp ${MBART_ROOT}/dict.zh_CN.txt ${MUSTC_v2_ROOT}/en-zh/spm_bpe250000_st.txt
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-zh/spm_bpe250000_asr.model
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-zh/spm_bpe250000_st.model
cp ${MBART_ROOT}/dict.en_XX.txt ${MUSTC_v2_ROOT}/en-ja/spm_bpe250000_asr.txt
cp ${MBART_ROOT}/dict.ja_JP.txt ${MUSTC_v2_ROOT}/en-ja/spm_bpe250000_st.txt
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-ja/spm_bpe250000_asr.model
cp ${MBART_ROOT}/sentence.bpe.model ${MUSTC_v2_ROOT}/en-ja/spm_bpe250000_st.model

# MuST-C v1 (en-de)
for task in {asr,st}; do
    python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_mustc_data.py \
    --data-root $MUSTC_v1_ROOT_AHC --task $task --use-audio-input \
    --only-manifest --append-lang-id --out-root $MUSTC_v1_ROOT \
    --vocab-type bpe --vocab-size 250000
done

# Europarl-ST (en-de)
for task in {asr,st}; do
    python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_europarl_data.py \
    -d ${EUROPARL_ROOT} --lang-pair en-de --task $task \
    --use-audio-input --only-manifest --append-lang-id \
    --out-root $EUROPARL_ROOT
done

# CoVoST (en-de,en-zh,en-ja)
for tgt_lang in {de,zh,ja}; do
    for task in {asr,st}; do
        python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_covost_data.py \
        -d $CV_ROOT -s en -t $tgt_lang --append-lang-id \
        --task $task --out-root $CV_ROOT --clips-name clips
    done
done

# Speech-Translation TED corpus (en-de)
for task in {asr,st}; do
    python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_stted_data.py \
    --data-root $ST_TED_ROOT_AHC --task $task --use-audio-input \
    --only-manifest --append-lang-id --out-root $ST_TED_ROOT \
    --vocab-type bpe --vocab-size 250000
done

# (2) Data Filtering
# (2-1) ASR inference on the "train" sets
# MuST-C
for tgt_lang in {de,ja,zh}; do
    python ${UPC_IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${MUSTC_v2_ROOT}/en-${tgt_lang}/train_asr.tsv -o ${MUSTC_v2_ROOT}/en
done

# MuST-C v1 (en-de; train data only)
python ${UPC_IWSLT_ROOT}/scripts/filtering/asr_inference.py \
--tsv_path ${MUSTC_v1_ROOT}/en-de/train_asr.tsv -o ${MUSTC_v1_ROOT}/en

# Europarl-ST
for split in {train,dev,test}; do
    python ${UPC_IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${EUROPARL_ROOT}/en/en-de_${split}_asr.tsv -o ${EUROPARL_ROOT}/en
done

# CoVoST
for split in {train,dev,test}; do
    for tgt_lang in {de,ja,zh}; do
        python ${UPC_IWSLT_ROOT}/scripts/filtering/asr_inference.py \
        --tsv_path ${COVOST_ROOT}/en-${tgt_lang}/${split}_asr.tsv -o ${COVOST_ROOT}/en
    done
done

# Speech-Translation TED corpus (en-de; train data only)
python ${UPC_IWSLT_ROOT}/scripts/filtering/asr_inference.py \
--tsv_path ${ST_TED_ROOT}/en-de/train_asr.tsv -o ${ST_TED_ROOT}/en

# (2-2) ASR-based and text-based filtering
# MuST-C
for tgt_lang in {de,ja,zh}; do
    python ${UPC_IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
    -tsv ${MUSTC_v2_ROOT}/en-${tgt_lang}/train_st.tsv \
    -p ${MUSTC_v2_ROOT}/en/train_asr_wer_results.json \
    -o ${MUSTC_v2_ROOT}/en-${tgt_lang} \
    -par -wer 0.75
done

# MuST-C v1 (en-de; train data only)
python ${UPC_IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
-tsv ${MUSTC_v1_ROOT}/en-de/train_st.tsv \
-p ${MUSTC_v1_ROOT}/en/train_asr_wer_results.json \
-o ${MUSTC_v1_ROOT}/en-de \
-par -wer 0.75

# Europarl-ST
for split in {train,dev,test}; do
    python ${UPC_IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
    -tsv ${EUROPARL_ROOT}/en/en-de_${split}_st.tsv \
    -p ${EUROPARL_ROOT}/en/en-de_${split}_asr_wer_results.json \
    -o ${EUROPARL_ROOT}/en \
    -par -wer 0.75
done

# CoVoST
for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        python ${UPC_IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
        -tsv ${COVOST_ROOT}/en-${tgt_lang}/${split}_st.tsv \
        -p ${COVOST_ROOT}/en/${split}_asr_wer_results.json \
        -o ${COVOST_ROOT}/en-${tgt_lang} \
        -par -wer 0.75
    done
done

# Speech-Translation TED corpus (en-de; train data only)
python ${UPC_IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
-tsv ${ST_TED_ROOT}/en-de/train_st.tsv \
-p ${ST_TED_ROOT}/en/train_asr_wer_results.json \
-o ${ST_TED_ROOT}/en-de \
-wer 0.75

# (3) Combining the different datasets into en-de, en-ja and en-zh directories
mkdir -p ${DATA_ROOT}/{en-de,en-zh,en-ja}
# from MuST-C
for tgt_lang in {de,ja,zh}; do
    ln -s ${MUSTC_v2_ROOT}/en-${tgt_lang}/config_st.yaml ${DATA_ROOT}/en-${tgt_lang}/config.yaml
    ln -s ${MUSTC_v2_ROOT}/en-${tgt_lang}/spm_bpe250000_st.{txt,model} $DATA_ROOT/en-${tgt_lang}/
    ln -s ${MUSTC_v2_ROOT}/en-${tgt_lang}/train_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_mustc.tsv
    ln -s ${MUSTC_v2_ROOT}/en-${tgt_lang}/dev_st.tsv ${DATA_ROOT}/en-${tgt_lang}/dev_mustc.tsv
    ln -s ${MUSTC_v2_ROOT}/en-${tgt_lang}/tst-COMMON_st.tsv ${DATA_ROOT}/en-${tgt_lang}/tst-COMMON_mustc.tsv
done

# from MuST-C v1
ln -s ${MUSTC_v1_ROOT}/en-de/train_st_filtered.tsv ${DATA_ROOT}/en-de/train_mustc_v1.tsv
ln -s ${MUSTC_v1_ROOT}/en-de/dev_st.tsv ${DATA_ROOT}/en-de/tran_dev_mustc_v1.tsv
ln -s ${MUSTC_v1_ROOT}/en-de/tst-COMMON_st.tsv ${DATA_ROOT}/en-de/train_tst-COMMON_mustc_v1.tsv

# from Europarl-ST
for split in {train,dev,test}; do
    if [[ $split != train ]]; then
        ln -s ${EUROPARL_ROOT}/en/en-de_${split}_st_filtered.tsv ${DATA_ROOT}/en-de/train_${split}_europarl.tsv
    else
        ln -s ${EUROPARL_ROOT}/en/en-de_${split}_st_filtered.tsv ${DATA_ROOT}/en-de/${split}_europarl.tsv
    fi
done

# from CoVoST
for tgt_lang in {de,ja,zh}; do
    for split in {train,dev,test}; do
        if [[ $split != train ]]; then
            ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/train_${split}_covost.tsv
        else
            ln -s ${COVOST_ROOT}/en-${tgt_lang}/${split}_st_filtered.tsv ${DATA_ROOT}/en-${tgt_lang}/${split}_covost.tsv
        fi
    done
done

# from Speech-Translation TED corpus
ln -s ${ST_TED_ROOT}/en-de/train_st_filtered.tsv ${DATA_ROOT}/en-de/train_stted.tsv


# (4) Fix size mismatch (250001 -> 250054). mBART50 was trained with 53 extra special tokens.
# ref: https://github.com/facebookresearch/fairseq/issues/3474
for tgt_lang in {de,ja,zh}; do
    cat ${MBART_ROOT}/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | \
      sed 's/$/> 1/g' >> ${DATA_ROOT}/en-${tgt_lang}/spm_bpe250000_st.txt && \
      echo "<mask> 1" >> ${DATA_ROOT}/en-${tgt_lang}/spm_bpe250000_st.txt
done
