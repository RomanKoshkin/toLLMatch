#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# Download Europarl-ST v1.1
# ref: https://github.com/mt-upc/iwslt-2021
mkdir -p ${EUROPARLST_ROOT} && \
wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O - | \
  tar -xz --strip-components 1 -C ${EUROPARLST_ROOT}
for f in ${EUROPARLST_ROOT}/*/audios/*.m4a; do
  ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.m4a}.wav" && rm $f
done

# Download the Common Voice version 8 and the CoVoST tsvs (en-de, en-ja, en-zh)
mkdir -p ${COVOST_ROOT}/{en-de,en-ja,en-zh}
wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-en.tar.gz
tar -xvf cv-corpus-8.0-2022-01-19-en.tar.gz -C $COVOST_ROOT
rm cv-corpus-8.0-2022-01-19-en.tar.gz
mkdir -p ${COVOST_ROOT}/{en-de,en-ja,en-zh}
tar -xvf ${COVOST_ROOT_AHC}/download/covost_v2.en_de.tsv.tar.gz -C ${COVOST_ROOT}/en-de
tar -xvf ${COVOST_ROOT_AHC}/download/covost_v2.en_zh-CN.tsv.tar.gz -C ${COVOST_ROOT}/en-zh
tar -xvf ${COVOST_ROOT_AHC}/download/covost_v2.en_ja.tsv.tar.gz -C ${COVOST_ROOT}/en-ja
for f in ${CV_ROOT}/*/clips/*.mp3; do
  ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.mp3}.wav" && rm $f
done
sed -i 's/\.mp3\t/\.wav\t/g' ${CV_ROOT}/**/*.tsv
sed -i 's/\.mp3\t/\.wav\t/g' ${COVOST_ROOT}/**/*.tsv

# Download the IWSLT data (tst.2019,tst.2020,tst.2021,tst.2022)
mkdir -p $IWSLT_TST_ROOT
for year in {2019,2020,2021}; do
    wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/IWSLT-SLT.tst${year}.en-de.tgz
    tar -xvf IWSLT-SLT.tst${year}.en-de.tgz -C ${IWSLT_TST_ROOT}
    rm IWSLT-SLT.tst${year}.en-de.tgz
done
for tgt_lang in {de,ja,zh}; do
    wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-${tgt_lang}/IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    tar -xvf IWSLT-SLT.tst2022.en-${tgt_lang}.tgz -C ${IWSLT_TST_ROOT}
    rm IWSLT-SLT.tst2022.en-${tgt_lang}.tgz
    # get the file order for this pair
    cut -d' ' -f1 ${IWSLT_TST_ROOT}/IWSLT.tst2022/IWSLT.TED.tst2022.en-${tgt_lang}.en.video_url > ${IWSLT_TST_ROOT}/IWSLT.tst2022/FILER_ORDER.en-${tgt_lang}
done
