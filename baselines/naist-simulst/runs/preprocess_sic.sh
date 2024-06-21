#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=$1

# (1) Data Preparation
# NAIST_SIC (en-ja)
python ${UPC_IWSLT_ROOT}/scripts/data_prep/prep_sic_data.py \
--data-root ${ROOT}/data/dataset/NAIST_SIC/data/mustc_format --task st --use-audio-input \
--only-manifest --append-lang-id --out-root ${ROOT}/data/dataset/NAIST_SIC/data/training \
--vocab-type bpe --vocab-size 250000

# (2) Data Filtering
# (2-1) ASR inference on the "train" sets
# (2-2) ASR-based and text-based filtering

# (3) Combining the different datasets into en-de, en-ja and en-zh directories
ln -s ${ROOT}/data/dataset/NAIST_SIC/data/training/train_st.tsv ${DATA_ROOT}/en-ja/train_sic.tsv
ln -s ${ROOT}/data/dataset/NAIST_SIC/data/training/dev_st.tsv ${DATA_ROOT}/en-ja/dev_sic.tsv
ln -s ${ROOT}/data/dataset/NAIST_SIC/data/training/test_st.tsv ${DATA_ROOT}/en-ja/test_sic.tsv

# (4) Prepare data for SimulEval
python ${UPC_IWSLT_ROOT}/scripts/data_prep/seg_sic_data.py \
--data-root ${ROOT}/data/dataset/NAIST_SIC/data/mustc_format \
--output ${ROOT}/data/eval_data/si/en-ja \
--split test
