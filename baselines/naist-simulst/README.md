
# NAIST's Speech Translation System for IWSLT 2023
<!-- This repository is based on [mt-upc/iwslt2022](https://github.com/mt-upc/iwslt-2022/tree/470605160226686b69213a45e1797cb918be0976). 
Original codes are stored at [upc@404706051](https://github.com/ahclab/iwslt-2023/tree/upc%404706051) branch. -->

NAIST's system submitted to the [IWSLT 2023 simultaneous track](https://iwslt.org/2023/simultaneous).

The paper is available [here](https://aclanthology.org/2023.iwslt-1.31.pdf).

## Abstract
This paper describes NAIST’s submission to the IWSLT 2023 Simultaneous Speech Translation task: English-to-{German, Japanese, Chinese} speech-to-text translation and English-to-Japanese speech-to-speech translation. Our speech-to-text system uses an end-to-end multilingual speech translation model based on large-scale pre-trained speech and text models. We add Inter-connections into the model to incorporate the outputs from intermediate layers of the pre-trained speech model and augment prefix-to-prefix text data using Bilingual Prefix Alignment to enhance the simultaneity of the offline speech translation model. Our speech-to-speech system employs an incremental text-to-speech module that consists of a Japanese pronuncia- tion estimation model, an acoustic model, and a neural vocoder.

# Setup
```
git clone --recursive git@github.com:ahclab/naist-simulst.git
pip install -r  requirements.txt
```

## ~~Download models and data (tmp)~~
These models are currently not publicly available. Training instructions will be published instead.
~~Download the files required for execution from the links below:~~
- ~~SimulS2T En-De: [en-de.tar.gz](https://drive.google.com/file/d/1C3C_3CWQvE-fJ3d5A2-90WTHOslXl-ei/view?usp=sharing)~~
- ~~SimulS2T En-Ja: [en-ja.tar.gz](https://drive.google.com/file/d/1mrMzsJazKtOwiy0_nh9wAG6gPIZklbcE/view?usp=sharing)~~
- ~~SimulS2T En-Zh: [en-zh.tar.gz](https://drive.google.com/file/d/1lEgzOWPQNtk-bS7mAHmsTmw9lpK46xjV/view?usp=sharing)~~
- ~~SimulS2S En-Ja: [en-ja-tts.tar.gz](https://drive.google.com/file/d/1G1hKMeFWLvgvszPgTskBnElhrJv2bu2Z/view?usp=sharing)~~
- ~~MuST-C evaluation data: [evaldata.tar.gz](https://drive.google.com/file/d/1eK74e30pwtiEe8pTSQk8kKPJbgdj0IFf/view?usp=sharing)~~

~~Before running inference, local paths in commands need to be replaced as follows:~~
- ~~Replace `/ahc/work3/sst-team/IWSLT2023/shared/en-*` with the path of unzipped `en-*.tar.gz`~~
- ~~Replace `/ahc/work3/sst-team/IWSLT2023/data/eval_data` with the path of unzipped `eval_data.tar.gz`~~
- ~~Replace `/ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts` with the path of unzipped `en-ja-tts.tar.gz`~~

# Inference
## [Private] Evaluation of SimulS2T En-De for MuST-C
```
OUTPUT_DIR=results/ende
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2t_la_word.py \
  --sentencepiece-model /ahc/work3/sst-team/IWSLT2023/shared/en-de/data-bin/spm_bpe250000_st.model \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-de/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-de/evaldata/tst-COMMON.de \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-de/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-de/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang de \
  --source-segment-size 950 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer 13a \
  --end-index 10
```

You can get a target text file "generation.txt" in $OUTPUT_DIR by running the following command: 
```
python scripts/simulst/log2gen.py ${OUTPUT_DIR}/instances.log
```

## [Private] Evaluation of SimulS2T En-Ja for MuST-C
```
OUTPUT_DIR=results/enja
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2t_la_char.py \
  --eval-latency-unit char --filtered-tokens '▁' \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.ja \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-ja/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-ja/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang ja \
  --source-segment-size 650 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer ja-mecab \
  --end-index 10
```

## [Private] Evaluation of SimulS2T En-Zh for MuST-C
```
OUTPUT_DIR=results/enzh
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2t_la_char.py \
  --eval-latency-unit char --filtered-tokens '▁' \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-zh/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-zh/evaldata/tst-COMMON.zh \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-zh/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-zh/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang zh \
  --source-segment-size 700 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer zh \
  --end-index 10
```

## [Private] Evaluation of SimulS2S En-Ja for MuST-C
```
OUTPUT_DIR=results/enja-s2s
TTS_MODELS_PATH=/ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts/tts_model
SUB2YOMI_PATH=${TTS_MODELS_PATH}/base_model1/sub2yomi/output0.out
YOMI2TTS_PATH=${TTS_MODELS_PATH}/base_model1/yomi2tts/checkpoint_100000.pth.tar
TTS2WAV_PATH=${TTS_MODELS_PATH}/base_model1/tts2wav/checkpoint_400000.pth.tar
SUB2YOMI_DICT_PATH=${TTS_MODELS_PATH}/base_model1/sub2yomi/vocabs_thd1.dict
YOMI2TTS_DICT_PATH=${TTS_MODELS_PATH}/base_model1/yomi2tts/pron.json
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2s_la_1_iwslt23.py \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.ja \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-ja/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-ja/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang ja \
  --source-segment-size 650 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer ja-mecab \
  --quality-metrics WHISPER_ASR_BLEU \
  --latency-metrics StartOffset EndOffset ATD \
  --target-speech-lang ja \
  --end-index 10 \
  --sub2yomi_model_path $SUB2YOMI_PATH \
  --yomi2tts_model_path $YOMI2TTS_PATH \
  --tts2wav_model_path $TTS2WAV_PATH \
  --sub2yomi_dict_path $SUB2YOMI_DICT_PATH \
  --yomi2tts_dict_path $YOMI2TTS_DICT_PATH
```

## [Private] Evaluation of SimulS2S En-Ja with ITTS (ver2)
RNN enc-dec pronunciation estimation (wait-k) + Tacotron2 + Parallel WaveGAN
A synthesis chunk is a morpheme unit
```
OUTPUT_DIR=results/enja-s2s-ver2
TTS_MODELS_PATH=/ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts/tts_model
SUB2YOMI_PATH=${TTS_MODELS_PATH}/base_model1/sub2yomi/output0.out
YOMI2TTS_PATH=${TTS_MODELS_PATH}/base_model1/yomi2tts/checkpoint_100000.pth.tar
TTS2WAV_PATH=${TTS_MODELS_PATH}/base_model1/tts2wav/checkpoint_400000.pth.tar
SUB2YOMI_DICT_PATH=${TTS_MODELS_PATH}/base_model1/sub2yomi/vocabs_thd1.dict
YOMI2TTS_DICT_PATH=${TTS_MODELS_PATH}/base_model1/yomi2tts/pron.json
simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2s_la_2_iwslt23.py \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.ja \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-ja/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-ja/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang ja \
  --source-segment-size 650 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer ja-mecab \
  --quality-metrics WHISPER_ASR_BLEU \
  --latency-metrics StartOffset EndOffset ATD \
  --target-speech-lang ja \
  --end-index 10 \
  --sub2yomi_model_path $SUB2YOMI_PATH \
  --yomi2tts_model_path $YOMI2TTS_PATH \
  --tts2wav_model_path $TTS2WAV_PATH \
  --sub2yomi_dict_path $SUB2YOMI_DICT_PATH \
  --yomi2tts_dict_path $YOMI2TTS_DICT_PATH
```

## [Private] Evaluation of SimulS2S En-Ja with ITTS (ver3)
RNN enc-dec pronunciation and accent info. estimation (wait-k)+ Tacotron2 + Parallel WaveGAN
A synthesis chunk is an accent phrase unit
```
OUTPUT_DIR=results/enja-s2s-ver3
TTS_MODELS_PATH=/ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts/tts_model
SUB2YOMI_PATH=${TTS_MODELS_PATH}/base_model3/sub2yomi/output0.out
YOMI2TTS_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/checkpoint_64000.pth.tar
TTS2WAV_PATH=${TTS_MODELS_PATH}/base_model3/tts2wav/checkpoint_400000.pth.tar
SUB2YOMI_DICT_PATH=${TTS_MODELS_PATH}/base_model3/sub2yomi/vocabs_thd1.dict
YOMI2TTS_DICT_P_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/phoneme.json
YOMI2TTS_DICT_A1_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a1.json
YOMI2TTS_DICT_A2_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a2.json
YOMI2TTS_DICT_A3_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/a3.json
YOMI2TTS_DICT_F1_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/f1.json
YOMI2TTS_DICT_F2_PATH=${TTS_MODELS_PATH}/base_model3/yomi2tts/f2.json

simuleval \
  --agent scripts/simulst/agents/v1.1.0/s2s_la_3_accent.py \
  --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.wav_list \
  --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.ja \
  --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-ja/checkpoint_best.pt \
  --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-ja/data-bin \
  --use-audio-input \
  --output $OUTPUT_DIR \
  --lang ja \
  --source-segment-size 650 \
  --la-n 2 \
  --beam 5 \
  --remote-port 2000 \
  --gpu \
  --sacrebleu-tokenizer ja-mecab \
  --quality-metrics WHISPER_ASR_BLEU \
  --latency-metrics StartOffset EndOffset ATD \
  --target-speech-lang ja \
  --end-index 10 \
  --sub2yomi_model_path $SUB2YOMI_PATH \
  --yomi2tts_model_path $YOMI2TTS_PATH \
  --tts2wav_model_path $TTS2WAV_PATH \
  --sub2yomi_dict_path $SUB2YOMI_DICT_PATH \
  --yomi2tts_phoneme_dict_path $YOMI2TTS_DICT_P_PATH \
  --yomi2tts_a1_dict_path $YOMI2TTS_DICT_A1_PATH \
  --yomi2tts_a2_dict_path $YOMI2TTS_DICT_A2_PATH \
  --yomi2tts_a3_dict_path $YOMI2TTS_DICT_A3_PATH \
  --yomi2tts_f1_dict_path $YOMI2TTS_DICT_F1_PATH \
  --yomi2tts_f2_dict_path $YOMI2TTS_DICT_F2_PATH \
```

## [Private] Evaluation of SimulS2S En-Ja with ITTS (ver4)
Transformer enc-dec pronunciation estimation (AlignATT) + Parallel acoustic model (Fastpitch like) + Parallel WaveGAN
A synthesis chunk depends on outputs from Transformer enc-dec pronunciation estimation
Two model (output0.out and output60000.out) may exsist below the link: /ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts/tts_model/base_model4/sub2yomi/
The latest model is output0.out, and you should use the latest one.
```
OUTPUT_DIR=results/enja-s2s-ver4
TTS_MODELS_PATH=/ahc/work3/sst-team/IWSLT2023/shared/en-ja-tts/tts_model
SUB2YOMI_PATH=${TTS_MODELS_PATH}/base_model4/sub2yomi/output0.out
YOMI2TTS_PATH=${TTS_MODELS_PATH}/base_model4/yomi2tts/checkpoint_100000.pth.tar
TTS2WAV_PATH=${TTS_MODELS_PATH}/base_model4/tts2wav/checkpoint_400000.pth.tar
SUB2YOMI_DICT_PATH=${TTS_MODELS_PATH}/base_model4/sub2yomi/vocabs_thd1.dict
YOMI2TTS_DICT_PHONEME_PATH=${TTS_MODELS_PATH}/base_model4/yomi2tts/phoneme.json
YOMI2TTS_DICT_PP_PATH=${TTS_MODELS_PATH}/base_model4/yomi2tts/phraseSymbol.json

simuleval \
   --agent scripts/simulst/agents/v1.1.0/s2s_la_4_transformer_average_alignatt.py \
   --source /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.wav_list \
   --target /ahc/work3/sst-team/IWSLT2023/data/eval_data/en-ja/evaldata/tst-COMMON.ja \
   --model-path /ahc/work3/sst-team/IWSLT2023/shared/en-ja/checkpoint_best.pt \
   --data-bin /ahc/work3/sst-team/IWSLT2023/shared/en-ja/data-bin \
   --use-audio-input \
   --output $OUTPUT_DIR \
   --lang ja \
   --source-segment-size 650 \
   --la-n 2 \
   --beam 5 \
   --remote-port 2000 \
   --gpu \
   --sacrebleu-tokenizer ja-mecab \
   --quality-metrics WHISPER_ASR_BLEU \
   --latency-metrics StartOffset EndOffset ATD \
   --target-speech-lang ja \
   --end-index 10 \
   --sub2yomi_model_path $SUB2YOMI_PATH \
   --yomi2tts_model_path $YOMI2TTS_PATH \
   --tts2wav_model_path $TTS2WAV_PATH \
   --sub2yomi_dict_path $SUB2YOMI_DICT_PATH\
   --yomi2tts_phoneme_dict_path $YOMI2TTS_PHONEME_DICT_PATH\
   --yomi2tts_pp_dict_path $YOMI2TTS_PP_DICT_PATH
```

# Docker images
You can download our submissions to IWSLT 2023 from here.
Each system contains a compressed docker image file `image.tar`.
Follow the `readme.md` to reproduce the results of the system paper. 
- IWSLT2023_NAIST
  - s2t_en-de
  - s2t_en-ja
  - s2t_en-zh
  - s2s_en-ja

# Others
The repository contains several other implementations:
- [EDAtt](https://arxiv.org/pdf/2212.07850.pdf) and [ALIGNATT](https://arxiv.org/pdf/2305.11408.pdf) policies
- [Morpheme-based TTS](scripts/simulst/agents/v1.1.0/s2s_la_2_morpheme.py), [Accent-based TTS](scripts/simulst/agents/v1.1.0/s2s_la_3_accent.py)
