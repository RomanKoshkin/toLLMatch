#!/bin/bash

ckpt_file=$1
test_set=$2     # IWSLT.tst2020, IWSLT.tst2019
tgt_lang=$3     # de, ja, zh
segm_yaml=$4

src_lang=en

eval "$(conda shell.bash hook)"
conda activate iwslt22

path_to_wavs=${IWSLT_DATA}/${test_set}/wavs

if [[ $tgt_lang == "de" ]]; then
  tokenizer=13a
elif [[ $tgt_lang == "ja" ]]; then
  tokenizer=ja-mecab
elif [[ $tgt_lang == "zh" ]]; then
  tokenizer=zh
fi

if [[ $test_set == IWSLT.tst2022 ]]; then
    file_order_name=FILE_ORDER_${tgt_lang}
else
    file_order_name=FILE_ORDER
fi

path_to_tsv=${IWSLT_DATA}/${test_set}/shas.tsv
echo "Preparing custom dataset tsv"
python ${IWSLT_ROOT}/scripts/segmentation/prep_custom_data.py \
    -yaml $segm_yaml \
    -wav $path_to_wavs \
    -order ${IWSLT_DATA}/${test_set}/${file_order_name} \
    -tsv $path_to_tsv \
    -lang $tgt_lang

echo "Generating translations"
fairseq-generate ${IWSLT_DATA}/${test_set} \
    --path $ckpt_file \
    --data-config-yaml ${DATA_ROOT}/${src_lang}-${tgt_lang}/config.yaml \
    --task speech_to_text \
    --gen-subset shas \
    --seed 48151623 \
    --prefix-size 1 \
    --scoring sacrebleu \
    --max-target-positions 1024 \
    --max-tokens 1_920_000 \
    --max-source-positions 1_500_000 \
    --beam 5 \
    --sacrebleu-tokenizer $tokenizer > ${IWSLT_DATA}/${test_set}/fairseq_generate.out 2>&1

echo "Formatting translations"
python ${IWSLT_ROOT}/scripts/segmentation/format_gen_output.py \
    -i ${IWSLT_DATA}/${test_set}/fairseq_generate.out \
    -o ${IWSLT_DATA}/${test_set}/fairseq_generate_formated.out

if [[ $test_set == IWSLT.tst2021 || $test_set == IWSLT.tst2022 ]]; then
    continue
fi

# activate the secondary python2 env
eval "$(conda shell.bash hook)"
conda activate snakes27

_set=$(echo $test_set| cut -d'.' -f 2)

# align the hypotheses with the references
echo "Aligning translations"
bash ${MWERSEGMENTER_ROOT}/segmentBasedOnMWER.sh \
    ${IWSLT_DATA}/${test_set}/IWSLT.TED.${_set}.${src_lang}-${tgt_lang}.${src_lang}.xml \
    ${IWSLT_DATA}/${test_set}/IWSLT.TED.${_set}.${src_lang}-${tgt_lang}.${tgt_lang}.xml \
    ${IWSLT_DATA}/${test_set}/fairseq_generate_formated.out \
    UPC \
    $tgt_lang \
    ${IWSLT_DATA}/${test_set}/fairseq_generate_formated_aligned.out \
    normalize \
    1

# re-activate main environment
eval "$(conda shell.bash hook)"
conda activate iwslt22

python ${IWSLT_ROOT}/scripts/segmentation/score_translation.py \
    -ref ${IWSLT_DATA}/${test_set}/__mreference \
    -trans ${IWSLT_DATA}/${test_set}/__segments \
    -s ${IWSLT_DATA}/${test_set}/results.json \
    -l $tgt_lang