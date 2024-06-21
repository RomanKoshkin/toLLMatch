import argparse
import pandas as pd
from typing import List

def get_split_name(filepath: str) -> str:
    filename = filepath.split("/")[-1]
    split_name = filename.split(".")[0]
    return split_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--n-shards",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--input-manifest-filepath",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    
    df = pd.read_csv(
        args.input_manifest_filepath,
        sep="\t",
    )
    
    n_lines = len(df.index)
    shard_n_lines = int(n_lines / args.n_shards)
    split_name = get_split_name(args.input_manifest_filepath)
    
    df_shard_list: List[pd.DataFrame] = []
    
    for i in range(args.n_shards):
        start_index = i * shard_n_lines
        if i-1 == args.n_shards:
            df_shard = df[start_index:]
        else:
            df_shard = df[start_index:start_index+shard_n_lines]
        df_shard_list.append(df_shard)
        print(df_shard)
    
    for i, df_shard in enumerate(df_shard_list):
        df_shard.to_csv(
            f"{args.shard_dir}/{split_name}_{i}.tsv",
            sep="\t",
            index=False,
        )

if __name__=="__main__":
    main()