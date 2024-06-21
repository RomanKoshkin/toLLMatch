# Prefix Alignment
We have implemented a script here to generate pairs of Prefix Alignment. The usage of the code is shown here.

## 1. Generate Prefix Alignment Pair
Go to `prefix_alignment/` and edit `prefix_alignment.sh`.
You can change the variables, depending on your GPU environment and data location.
Then run the script with the following arguments.
```bash
bash prefix_alignment.sh $split $gpu
```

## 2. Data Filtering
You can conduct data filetering using implemented script.
Data filtering determines the maximum value of the ratio calculated by `src_length / hyp_length` and removes data that exceeds the maximum value.
To perform data filtering, run the following.
```bash
bash data_filtering.sh
```

## 3. Detokenize Translation Pair
If necessary, you can detokenize the data. Edit the `detokenize.sh` and run the following.
```bash
bash detokenize.sh $split
```
