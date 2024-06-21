export ROOT=/ahc/work/sst-team/IWSLT2023
export UPC_IWSLT_ROOT=$(cd $(dirname $0)/..; pwd)

# tools
export FAIRSEQ_ROOT=${UPC_IWSLT_ROOT}/fairseq
export SCRIPT_ROOT=${UPC_IWSLT_ROOT}/scripts

# models
export MODELS_ROOT=${ROOT}/models
export MBART_ROOT=${MODELS_ROOT}/mbart50.ft.1n
export SAVE_DIR=${UPC_IWSLT_ROOT}/experiments

# data (AHC resources)
export MUSTC_v2_ROOT_AHC=/ahc/ahcshare/Data/MuST-C/v2.0_IWSLT2022
export MUSTC_v1_ROOT_AHC=/ahc/ahcshare/Data/MuST-C/v1.0
export COVOST_ROOT_AHC=/ahc/data/CoVoST
export ST_TED_ROOT_AHC=/ahc/data/Speech-Translation_TED_corpus/iwslt-corpus

# data
export FILTERING_ROOT=${ROOT}/data/dataset
export IWSLT_TST_ROOT=${ROOT}/data/dataset/IWSLT-SLT
export MUSTC_v2_ROOT=${ROOT}/data/dataset/MuST-C_v2
export MUSTC_v1_ROOT=${ROOT}/data/dataset/MuST-C_v1
export CV_ROOT=${ROOT}/data/dataset/CommonVoice_v8
export COVOST_ROOT=${CV_ROOT}/en/CoVoST
export ST_TED_ROOT=${ROOT}/data/dataset/ST_TED
export EUROPARL_ROOT=${ROOT}/data/dataset/Europarl-ST_v1.1
export DATA_ROOT=${ROOT}/data/training

export TTS_MODELS_PATH=/ahc/work3/sst-team/DemoSystem/2022/tomoya-ya/IWSLT2023_demo_open/work/tts_model
