SCRIPT_PATH=$(cd $(dirname $0)/..; pwd)/scripts/concat_itts_data.py

python $SCRIPT_PATH \
    --log-path /ahc/work/sst-team/IWSLT2023/work/yuta-nis/concat_itts/instances_timestamp.log \
    --output-dir /ahc/work/sst-team/IWSLT2023/work/yuta-nis/concat_itts_v2/
