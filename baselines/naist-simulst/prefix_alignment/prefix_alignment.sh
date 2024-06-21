#/!bin/bash

#
# You can run this script by `bash prefix_alignment.sh $split $gpu`
#

# You should edit here to specify data and model
split=$1
manifest_dir=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all
script_dir=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/iwslt-2023/prefix_alignment/scripts
model_path=/ahc/work/sst-team/IWSLT2023/work/ryo-fu/upc-iwslt-2022/experiments/full_corpora_st-langid-interconnection/from_ave5,freeze/ckpts/checkpoint_best.pt


# configuration for corpus generation
# this config is optimized for NVIDIA A100 GPU 40GB
seed=48151623
beam=5
max_source_positions=900_000
max_target_positions=1_024
max_tokens=1_000_000
sampling_rate=16000
prefix_duration=1

########################
# DON'T EDIT FROM HERE #
########################
temporal_dir=${manifest_dir}/tmp
input_manifest_path=${manifest_dir}/${split}.tsv
prefix_manifest_path=${manifest_dir}/${split}_prefix.tsv

pre_decision_size=$((sampling_rate * prefix_duration))
pre_decision_size=${pre_decision_size%.*}

if [ ! -e $temporal_dir ]; then
    mkdir -p $temporal_dir
fi


cmd="CUDA_VISIBLE_DEVICES=$2 python $script_dir/generate_prefix_translation_pair.py $manifest_dir
--path $model_path
--task speech_to_label
--gen-subset $split
--seed $seed
--prefix-size 1
--batch-size 1
--num-workers 2
--skip-invalid-size-inputs-valid-test
--no-progress-bar
--max-source-positions $max_source_positions
--max-target-positions $max_target_positions
--max-tokens $max_tokens
--pre-decision-size $pre_decision_size
--output-filepath $prefix_manifest_path
--beam $beam"
eval $cmd
