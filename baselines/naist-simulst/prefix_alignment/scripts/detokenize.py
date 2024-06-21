import argparse
import pandas as pd
import sentencepiece
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-filepath",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sentencepiece-modelpath",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output-manifest-filepath",
        type=str,
    )
    args = parser.parse_args()
    
    df_manifest = pd.read_csv(
        args.manifest_filepath,
        sep="\t",
    )
    
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(args.sentencepiece_modelpath)
    
    dict_manifest = df_manifest.to_dict("index")
    for manifest in tqdm(dict_manifest.values()):
        tgt_tokens = manifest["tgt_text"].split(" ")
        manifest["tgt_text"] = sp.DecodePieces(tgt_tokens)
    
    df_manifest = pd.DataFrame.from_dict(dict_manifest, orient="index")
    df_manifest.to_csv(
        args.output_manifest_filepath,
        sep="\t",
        index=False,
    )
    


if __name__=="__main__":
    main()