split=$1

manifest_dir=/ahc/work/sst-team/IWSLT2023/work/yuta-nis/dump2023/en-all
n_samples=2000

manifest_filepath=${manifest_dir}/${split}.tsv
output_filepath=${manifest_dir}/${split}_reduced.tsv

python scripts/reduce_corpus.py \
    --manifest-filepath $manifest_filepath \
    --output-manifest-filepath $output_filepath \
    --n-samples $n_samples