import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-ratio",
        default=3000,
        type=int,
    )
    parser.add_argument(
        "--input-filepath",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output-filepath",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    
    df_manifest = pd.read_csv(
        args.input_filepath,
        sep="\t",
    )
    manifest_length = len(df_manifest.index)
    manifest_columns = df_manifest.columns
    
    drop_count = 0
    filtered_manifest = []
    
    for manifest in tqdm(df_manifest.values, total=manifest_length):
        tgt_tokens = manifest[3].split(" ")
        token_length = len(tgt_tokens)
        n_frames = manifest[2]
        
        ratio = n_frames / token_length
        if ratio <= args.max_ratio:
            filtered_manifest.append(manifest)
        else:
            drop_count += 1
    
    drop_ratio = drop_count / manifest_length
    
    print(f"{drop_ratio*100:.2f}% data has been removed.")
    
    df_manifest = pd.DataFrame(filtered_manifest, columns=manifest_columns)
    df_manifest.to_csv(
        args.output_filepath,
        sep="\t",
        index=False,
    )
        

if __name__=="__main__":
    main()