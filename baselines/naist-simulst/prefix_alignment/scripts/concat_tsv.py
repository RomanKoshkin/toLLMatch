import pandas as pd
import argparse
import glob


def xor(x:bool, y:bool) -> bool:
    return bool((x and not y) or (not x and y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--manifest-filenames-regex",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--manifest-filenames",
        default=None,
        nargs="*",
        type=str,
    )
    parser.add_argument(
        "--output-filepath",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    
    assert xor((args.manifest_filenames_regex is None), (args.manifest_filenames is None))
    
    if args.manifest_filenames_regex is not None:
        manifest_filepaths_list = glob.glob(f"{args.manifest_dir}/{args.manifest_filenames_regex}")
    else:
        manifest_filepaths_list = [
            f"{args.manifest_dir}/{manifest_filename}" for manifest_filename in args.manifest_filenames
        ]
    
    # retrieve df info
    dfs = []
    for manifest_filepath in manifest_filepaths_list:
        print(f"Reading {manifest_filepath}...")
        df = pd.read_csv(
            manifest_filepath,
            sep="\t",
        )
        dfs.append(df)
        
        
    df_concat = pd.concat(dfs)
    df_concat.to_csv(
        args.output_filepath,
        sep="\t",
        index=False,
    )
    
    print(f"Manifest files (total {len(manifest_filepaths_list)} files) are combined into {args.output_filepath}")


if __name__=="__main__":
    main()