import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def simple_filter(args):
    src=args.source
    tgt=args.target
    out_dir=args.out_dir
    ratio=args.ratio_threshold

    with open(src) as f:
        arr_src = [line.strip() for line in f]
    with open(tgt) as f:
        arr_tgt = [line.strip() for line in f]

    src_path,src_code= os.path.splitext(src)
    tgt_path,tgt_code= os.path.splitext(tgt)
    if src_code is tgt_code:
        print('Don\'t use same language code.')
        exit()
    df_src=pd.DataFrame(arr_src,columns=[ src_code ])
    df_tgt=pd.DataFrame(arr_tgt,columns=[ tgt_code ])
    df=df_src.join(df_tgt)
    df['ratio'] = None
    print("processing data")
    df['ratio']=df.progress_apply(lambda x:len(x[src_code].split())/len(x[tgt_code].split()),axis=1)

    new_df=df[(df['ratio'] > 1/ratio) & (df['ratio'] < ratio)]
    print("saving cleaned"+src_code)
    np.savetxt(out_dir+'/cleaned'+src_code, new_df[src_code].to_numpy(), fmt='%s')
    print("saving cleaned"+tgt_code)
    np.savetxt(out_dir+'/cleaned'+tgt_code, new_df[tgt_code].to_numpy(), fmt='%s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="path to the source txt",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="path to the target txt",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        required=True,
        help="path to the save cleaned txt",
    )
    parser.add_argument(
        "--ratio-threshold",
        "-r",
        type=float,
        default=1.3,
        help="ratio threshould between source and target (default is 1.3)",
    )
    args = parser.parse_args()

    simple_filter(args)
