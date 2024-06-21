#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

# prepare parallel MT data for IWSLT2023
# ref: 
# https://iwslt.org/2022/offline
# https://www.statmt.org/wmt22/translation-task.html#download

src=en

# 1. data download (not exist in AHC)
# wikimatrix
export WIKIMATRIX_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/WikiMatrix; mkdir -p ${WIKIMATRIX_ROOT}
cd ${WIKIMATRIX_ROOT}/
for lang in {de-en,en-ja,en-zh}; do
    wget https://data.statmt.org/wmt21/translation-task/WikiMatrix/WikiMatrix.v1.${lang}.langid.tsv.gz -O WikiMatrix.v1.${lang}.langid.tsv.gz --no-check-certificate
    gzip -d WikiMatrix.v1.${lang}.langid.tsv.gz
done
cd -
# wikititles
export WIKITITLE_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/WikiTitles; mkdir -p ${WIKITITLE_ROOT}
cd ${WIKITITLE_ROOT}
for lang in {de-en,ja-en,zh-en}; do
    wget https://data.statmt.org/wikititles/v3/wikititles-v3.${lang}.tsv --no-check-certificate
done
cd -
# news commentary v16 (WMT21)
export NEWS_COMMENTARY_V16_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/News_Commentary_v16; mkdir -p ${NEWS_COMMENTARY_V16_ROOT}
cd ${NEWS_COMMENTARY_V16_ROOT}
for lang in {de-en,en-ja,en-zh}; do
    wget https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.${lang}.tsv.gz --no-check-certificate
    gzip -d news-commentary-v16.${lang}.tsv.gz
done
cd -
# KFTT + TED + JESC (in MTNT)
export KFTT_TED_JESC_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/KFTT_TED_JESC; mkdir -p ${KFTT_TED_JESC_ROOT}
cd ${KFTT_TED_JESC_ROOT}
wget https://github.com/pmichel31415/mtnt/releases/download/v1.0/clean-data-en-ja.tar.gz --no-check-certificate
tar -xvf clean-data-en-ja.tar.gz
cd -
# Common crawl (En-De) (WMT21)
export COMMON_CRAWL_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/Common_Crawl; mkdir -p ${COMMON_CRAWL_ROOT}
cd ${COMMON_CRAWL_ROOT}
wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz --no-check-certificate
tar -xvf training-parallel-commoncrawl.tgz
cd -

# tilde rapid corpus 2019 (en-de) (WMT21)
TILDE_RAPID_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/Tilde_Rapid; mkdir -p ${TILDE_RAPID_ROOT}
cd ${TILDE_RAPID_ROOT}
wget http://data.statmt.org/wmt20/translation-task/rapid/RAPID_2019.de-en.xlf.gz --no-check-certificate
gzip -d RAPID_2019.de-en.xlf.gz
cd -

# tilde model corpus (en-de) (WMT22) (skip)

# europal v10 (en-de) (WMT22)
export EUROPAL_v10_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/europal/v10; mkdir -p ${EUROPAL_v10_ROOT}
cd ${EUROPAL_v10_ROOT}/
wget https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz --no-check-certificate
gzip -d europarl-v10.de-en.tsv.gz
cd -

# paracrawl v9 (en-de, en-zh, en-ja) (WMT22)
export PARACRAWL_WMT22_ROOT_AHC=/ahc/data/ParaCrawl/v9

# opensubtitle2018
export OPENSUBTITLE_2018_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/OpenSubtitles2018/tmx; mkdir -p ${OPENSUBTITLE_2018_ROOT}
cd ${OPENSUBTITLE_2018_ROOT}
if [ ${tgt} = "zh" ]; then
    if [ ! -e en-zh_cn.tmx.gz ]; then
        wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/tmx/en-zh_cn.tmx.gz -O en-zh_cn.tmx.gz --no-check-certificate
        gzip -d en-zh_cn.tmx.gz 
    fi
elif [ ${tgt} = "ja" ]; then
    if [ ! -e en-ja.tmx.gz ]; then
        wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/tmx/en-ja.tmx.gz -O en-ja.tmx.gz --no-check-certificate
        gzip -d en-ja.tmx.gz 
    fi
elif [ ${tgt} = "de" ]; then
    if [ ! -e en-de.tmx.gz ]; then
        wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/tmx/de-en.tmx.gz -O de-en.tmx.gz --no-check-certificate
        gzip -d de-en.tmx.gz 
    fi
fi
cd -

# jesc (en-ja)
export JESC_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/JESC; mkdir ${JESC_ROOT}
cd ${JESC_ROOT}
wget https://nlp.stanford.edu/projects/jesc/data/split.tar.gz --no-check-certificate
tar xvzf split.tar.gz
cd -


# 2. parallel data prepare
for tgt in de ja zh; do
    lang=${src}-${tgt}
    MT_DATADIR=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/training/${lang}/mt/v1; mkdir -p ${MT_DATADIR}
    # wikimatrix
    export WIKIMATRIX_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/WikiMatrix; mkdir -p ${WIKIMATRIX_ROOT}
    if [ ${tgt} = "de" ]; then
        cut -f 2 ${WIKIMATRIX_ROOT}/WikiMatrix.v1.de-en.langid.tsv > ${MT_DATADIR}/train.wikimatrix.de
        cut -f 3 ${WIKIMATRIX_ROOT}/WikiMatrix.v1.de-en.langid.tsv > ${MT_DATADIR}/train.wikimatrix.en
    else
        cut -f 2 ${WIKIMATRIX_ROOT}/WikiMatrix.v1.${lang}.langid.tsv > ${MT_DATADIR}/train.wikimatrix.${src}
        cut -f 3 ${WIKIMATRIX_ROOT}/WikiMatrix.v1.${lang}.langid.tsv > ${MT_DATADIR}/train.wikimatrix.${tgt}
    fi

    # wikititles
    export WIKITITLE_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/WikiTitles; mkdir -p ${WIKITITLE_ROOT}
    # wikititle
    WIKITITLE_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/WikiTitles
    cut -f 1 ${WIKITITLE_ROOT}/wikititles-v3.${tgt}-${src}.tsv > ${MT_DATADIR}/train.wikititles.${tgt}
    cut -f 2 ${WIKITITLE_ROOT}/wikititles-v3.${tgt}-${src}.tsv > ${MT_DATADIR}/train.wikititles.${src}

    # news commentary v16 (WMT21)
    export NEWS_COMMENTARY_V16_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/News_Commentary_v16; mkdir -p ${NEWS_COMMENTARY_V16_ROOT}
    if [ ${tgt} = "de" ]; then
        cut -f 1 ${NEWS_COMMENTARY_V16_ROOT}/news-commentary-v16.de-en.tsv > ${MT_DATADIR}/train.newscom.v16.de
        cut -f 2 ${NEWS_COMMENTARY_V16_ROOT}/news-commentary-v16.de-en.tsv > ${MT_DATADIR}/train.newscom.v16.en
    else
        cut -f 1 ${NEWS_COMMENTARY_V16_ROOT}/news-commentary-v16.${lang}.tsv > ${MT_DATADIR}/train.newscom.v16.${src}
        cut -f 2 ${NEWS_COMMENTARY_V16_ROOT}/news-commentary-v16.${lang}.tsv > ${MT_DATADIR}/train.newscom.v16.${tgt}
    fi

    # opensubtitle2018
    export OPENSUBTITLE_2018_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/OpenSubtitles2018/tmx; mkdir -p ${OPENSUBTITLE_2018_ROOT}

    if [ ${tgt} = "zh" ]; then
        cat ${OPENSUBTITLE_2018_ROOT}/en-zh_cn.tmx | grep '<tuv xml:lang="en">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.en
        cat ${OPENSUBTITLE_2018_ROOT}/en-zh_cn.tmx | grep '<tuv xml:lang="zh_cn">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.zh
    elif [ ${tgt} = "ja" ]; then
        cat ${OPENSUBTITLE_2018_ROOT}/en-ja.tmx | grep '<tuv xml:lang="en">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.en
        cat ${OPENSUBTITLE_2018_ROOT}/en-ja.tmx | grep '<tuv xml:lang="ja">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.ja
    elif [ ${tgt} = "de" ]; then
        cat ${OPENSUBTITLE_2018_ROOT}/de-en.tmx | grep '<tuv xml:lang="en">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.en
        cat ${OPENSUBTITLE_2018_ROOT}/de-en.tmx | grep '<tuv xml:lang="de">' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/train.opus2018.de
    fi

    # ted iwslt2017
    export TED_ROOT_AHC=/ahc/ahcshare/Data/IWSLT/2017-01-trnted/texts
    export TED_IWSLT2017_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/TED_IWSLT2017; mkdir -p ${TED_IWSLT2017_ROOT}
    if [ -e ${TED_IWSLT2017_ROOT}/${lang} ]; then
        cp ${TED_ROOT_AHC}/${src}/${tgt}/${lang}.tgz ${TED_IWSLT2017_ROOT}/
        cd ${TED_IWSLT2017_ROOT}/; tar -xzvf ${lang}.tgz; cd -
    fi
    # only remain lines w/o tags
    cat ${TED_IWSLT2017_ROOT}/${lang}/train.tags.${lang}.${src}  | sed '/^</d' | sed -e 's/^ *//' > ${MT_DATADIR}/train.ted.${src}
    cat ${TED_IWSLT2017_ROOT}/${lang}/train.tags.${lang}.${tgt}  | sed '/^ </d' | sed -e 's/^ *//' > ${MT_DATADIR}/train.ted.${tgt} # tgtは最初に空白があるので正規表現が微妙に違う

    if [ ${tgt} = "ja" ]; then
        # KFTT + TED + JESC (in MTNT)
        export KFTT_TED_JESC_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/KFTT_TED_JESC; mkdir -p ${KFTT_TED_JESC_ROOT}
        cp ${KFTT_TED_JESC_ROOT}/train.${src} ${MT_DATADIR}/train.kftt-ted-jesc.${src}
        cp ${KFTT_TED_JESC_ROOT}/train.${tgt} ${MT_DATADIR}/train.kftt-ted-jesc.${tgt}
        
        # jesc (en-ja)
        export JESC_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/JESC/split
        cut -f 1 ${JESC_ROOT}/train > ${MT_DATADIR}/train.jesc.en
        cut -f 2 ${JESC_ROOT}/train > ${MT_DATADIR}/train.jesc.ja

        # kftt
        export KFTT_ROOT_AHC=/project/nakamura-lab01/Share/Corpora/ParaText/KFTT/data/orig/
        cp ${KFTT_ROOT_AHC}/kyoto-train.en ${MT_DATADIR}/train.kftt.en
        cp ${KFTT_ROOT_AHC}/kyoto-train.ja ${MT_DATADIR}/train.kftt.ja

        # jparacrawl v3 (en-ja) (WMT22, KIT)
        export JPARACRAWL_ROOT_AHC=/ahc/data/JParaCrawl/v3.0/${src}-${tgt}/
        cut -f 4 ${JPARACRAWL_ROOT_AHC}/en-ja.bicleaner05.txt > ${MT_DATADIR}/train.jparacrawl.v3.${src}
        cut -f 5 ${JPARACRAWL_ROOT_AHC}/en-ja.bicleaner05.txt > ${MT_DATADIR}/train.jparacrawl.v3.${tgt}

    fi

    if [ ${tgt} = "de" ]; then
        # Common crawl (En-De) (WMT21)
        export COMMON_CRAWL_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/Common_Crawl; mkdir -p ${COMMON_CRAWL_ROOT}
        cp ${COMMON_CRAWL_ROOT}/commoncrawl.${tgt}-${src}.${src} ${MT_DATADIR}/train-commoncrawl.${src}
        cp ${COMMON_CRAWL_ROOT}/commoncrawl.${tgt}-${src}.${tgt} ${MT_DATADIR}/train-commoncrawl.${tgt}

        # tilde rapid corpus 2019 (en-de) (WMT21)
        TILDE_RAPID_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/Tilde_Rapid; mkdir -p ${TILDE_RAPID_ROOT}
        cat ${TILDE_RAPID_ROOT}/RAPID_2019.${tgt}-${src}.xlf | grep '<source xml:lang="de">' | sed -e 's/<[^>]*>//g' > ${MT_DATADIR}/train.rapid2019.de
        cat ${TILDE_RAPID_ROOT}/RAPID_2019.${tgt}-${src}.xlf | grep '<target xml:lang="en">' | sed -e 's/<[^>]*>//g' > ${MT_DATADIR}/train.rapid2019.en

        # europal v10 (en-de) (WMT22)
        export EUROPAL_v10_ROOT=/ahc/work/sst-team/IWSLT2023/work/yuka-ko/data/dataset/europal/v10; mkdir -p ${EUROPAL_v10_ROOT}
        cut -f 1 ${EUROPAL_v10_ROOT}/europarl-v10.de-en.tsv > ${MT_DATADIR}/train.europal.v10.de
        cut -f 2 ${EUROPAL_v10_ROOT}/europarl-v10.de-en.tsv > ${MT_DATADIR}/train.europal.v10.en
        # paracrawl v9 (en-de, en-zh) (WMT22)
        export PARACRAWL_WMT22_ROOT_AHC=/ahc/data/ParaCrawl/v9
        cut -f 1 ${PARACRAWL_WMT22_ROOT_AHC}/en-de.txt > ${MT_DATADIR}/train.paracrawl.v9.${src}
        cut -f 2 ${PARACRAWL_WMT22_ROOT_AHC}/en-de.txt > ${MT_DATADIR}/train.paracrawl.v9.${tgt}

    fi
    if [ ${tgt} = "zh" ]; then
        # paracrawl v9 (en-de, en-zh) (WMT22)
        cut -f 1 ${PARACRAWL_WMT22_ROOT_AHC}/en-zh-v1.txt > ${MT_DATADIR}/train.paracrawl.v9.${src}
        cut -f 2 ${PARACRAWL_WMT22_ROOT_AHC}/en-zh-v1.txt > ${MT_DATADIR}/train.paracrawl.v9.${tgt}
    fi

    # dev and test (anyway TED_IWSLT2017 dev, test)
    rm ${MT_DATADIR}/dev.ted.${src}
    rm ${MT_DATADIR}/dev.ted.${tgt}
    for split in dev2010 tst2010 tst2011 tst2012 tst2013 tst2014 tst2015; do
        cat ${TED_IWSLT2017_ROOT}/${lang}/IWSLT17.TED.${split}.${lang}.${src}.xml | grep '<seg id' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/dev.${split}.${src}
        cat ${MT_DATADIR}/dev.${split}.${src} >> ${MT_DATADIR}/dev.ted.${src}
        cat ${TED_IWSLT2017_ROOT}/${lang}/IWSLT17.TED.${split}.${lang}.${tgt}.xml | grep '<seg id' | sed -e 's/<[^>]*>//g' -e 's/^ *//' > ${MT_DATADIR}/dev.${split}.${tgt}
        cat ${MT_DATADIR}/dev.${split}.${tgt} >> ${MT_DATADIR}/dev.ted.${tgt}
    done
done

