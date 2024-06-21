
import argparse
import os
import fasttext
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# supported lids
#af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
def lid_filter(args):
    src=args.source
    tgt=args.target
    path=args.model_path
    out_dir=args.out_dir
    model=fasttext.load_model(path)
    with open(src) as f:
        arr_src = [line.strip() for line in f]
    with open(tgt) as f:
        arr_tgt = [line.strip() for line in f]

    _,src_code= os.path.splitext(src)
    _,tgt_code= os.path.splitext(tgt)
    df_src=pd.DataFrame(arr_src,columns=[ src_code ])
    df_tgt=pd.DataFrame(arr_tgt,columns=[ tgt_code ])
    df=df_src.join(df_tgt)
    df['LID_src'] = None
    df['LID_tgt'] = None
    
    print("processing data")
    f = lambda x:list(model.predict(x[src_code])[0])[0][len('__label__'):]
    df['LID_src']=df.progress_apply(f,axis=1)
    f = lambda x:list(model.predict(x[tgt_code])[0])[0][len('__label__'):]
    df['LID_tgt']=df.progress_apply(f,axis=1)
    new_df=df[(df['LID_src'] == src_code[1:]) & (df['LID_tgt'] == tgt_code[1:])]
    print("saving file...")
    np.savetxt(out_dir+'/cleaned'+src_code, new_df[src_code].to_numpy(), fmt='%s')
    np.savetxt(out_dir+'/cleaned'+tgt_code, new_df[tgt_code].to_numpy(), fmt='%s')
    print("saved to "+ path)

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
        "--model_path",
        "-path",
        type=str,
        required=True,
        help="path to the lid model",
    )

    args = parser.parse_args()

    lid_filter(args)