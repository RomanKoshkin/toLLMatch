split=$1

manifest_dir=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all
tokenizer_path=/ahc/work/sst-team/IWSLT2023/data/dataset/MuST-C_v2/en-de/spm_bpe250000_st.model

manifest_filepath=${manifest_dir}/${split}.tsv
output_filepath=${manifest_dir}/${split}_detokenized.tsv

python scripts/detokenize.py \
    --manifest-filepath $manifest_filepath \
    --output-manifest-filepath $output_filepath \
    --sentencepiece-modelpath $tokenizer_path