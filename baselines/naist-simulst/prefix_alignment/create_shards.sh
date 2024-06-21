shard_dir=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all/shards
n_shards=8
input_manifest_path=/ahc/work/sst-team/IWSLT2023/data/training/en-ja/train_sic.tsv

if [ ! -e $shard_dir ]; then
    mkdir -p $shard_dir
fi

cmd="python scripts/generate_manifest_shards.py
    --shard-dir $shard_dir
    --n-shards $n_shards
    --input-manifest-filepath $input_manifest_path"
eval $cmd