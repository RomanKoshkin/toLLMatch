python scripts/concat_tsv.py \
    --manifest-dir /ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all \
    --manifest-filenames \
    train_mustc_ja_prefix_filtered_max4000_detokenized_tag.tsv \
    train_mustc_de_prefix_filtered_max4000_detokenized_tag.tsv \
    train_mustc_zh_prefix_filtered_max4000_detokenized_tag.tsv \
    --output-filepath /ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all/train_mustc_all_prefix_filtered_max4000_detokenized_tag.tsv
