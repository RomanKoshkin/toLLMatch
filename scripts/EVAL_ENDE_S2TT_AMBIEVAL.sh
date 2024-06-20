#!/bin/bash

ASR_MODEL=distil-large-v3
echo "ASR model: $ASR_MODEL"

echo "\n******** 🔥🔥 ${red}Make sure that vLLM runs${reset} ${green} $MODEL_ID${reset} 🔥🔥 ********\n"
sleep 2

FUNC_WORDS_LIST=(
  "_"
)

ID=0
END_INDEX=2

cd ../evaluation
find ./ONLINE_TARGETS -type d -name "out_*" -exec rm -rf {} +
for FUNC_WORDS in "${FUNC_WORDS_LIST[@]}"; do
    for K in 1; do
        for MIN_READ_TIME in 0.6 1.2 ; do
            simuleval \
                --source SOURCES/ambiguity_en \
                --target OFFLINE_TARGETS/ambiguity_de.txt \
                --background BACKGROUND_INFO/ambiguity_en_de.txt \
                --agent s2tt_agent.py \
                --k $K \
                --output ONLINE_TARGETS/out_${0}_${ID} \
                --start-index 0 \
                --end-index $END_INDEX \
                --verbose \
                --dir en-de \
                --use_api \
                --latency-metrics LAAL AL AP DAL \
                --quality-metrics BLEU CHRF \
                --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                --source-segment-size 200 \
                --min_read_time $MIN_READ_TIME \
                --use_asr_api \
                --asr_model_size $ASR_MODEL \
                --min_lag_words 1 \
                --prompt_id 1 \
                --func_wrds $FUNC_WORDS \
                --priming
            ID=$((ID+1))

            simuleval \
                --source SOURCES/ambiguity_en \
                --target OFFLINE_TARGETS/ambiguity_de.txt \
                --agent s2tt_agent.py \
                --k $K \
                --output ONLINE_TARGETS/out_${0}_${ID} \
                --start-index 0 \
                --end-index $END_INDEX \
                --verbose \
                --dir en-de \
                --use_api \
                --latency-metrics LAAL AL AP DAL \
                --quality-metrics BLEU CHRF \
                --model_id meta-llama/Meta-Llama-3-70B-Instruct \
                --source-segment-size 200 \
                --min_read_time $MIN_READ_TIME \
                --use_asr_api \
                --asr_model_size $ASR_MODEL \
                --min_lag_words 1 \
                --prompt_id 1 \
                --func_wrds $FUNC_WORDS \
                --priming
            ID=$((ID+1))
        done
    done
done