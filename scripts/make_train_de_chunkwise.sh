#!/bin/bash
STEP=1000
LIM=20000
DEVICE_ID=0

for FROM in $(seq 0 $STEP $((LIM-STEP))); do
    echo $FROM $((FROM+STEP))
    python align_data_chunkwise.py $FROM $((FROM+STEP)) $DEVICE_ID &
done