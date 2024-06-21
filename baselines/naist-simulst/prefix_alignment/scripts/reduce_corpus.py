import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-filepath",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-manifest-filepath",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--reduce-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    
    assert (args.n_samples is None and args.reduce_ratio is not None) or (args.n_samples is not None and args.reduce_ratio is None)
    
    df_manifest = pd.read_csv(
        args.manifest_filepath,
        sep="\t",
    )
    df_manifest = df_manifest.sample(args.n_samples, frac=args.reduce_ratio, random_state=0)
    df_manifest.to_csv(
        args.output_manifest_filepath,
        sep="\t",
        index=False,
    )
    


if __name__=="__main__":
    main()