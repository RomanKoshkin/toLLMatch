max_ratio=3200
prefix_manifest_path=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all/dev_mustc_de_ja_zh_prefix.tsv
filtered_manifest_path=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all/dev_mustc_de_ja_zh_prefix_filtered_max${max_ratio}.tsv

cmd="python scripts/data_filtering.py
    --max-ratio $max_ratio
    --input-filepath $prefix_manifest_path
    --output-filepath $filtered_manifest_path"
eval $cmd

echo "saved to ${filtered_manifest_path}"