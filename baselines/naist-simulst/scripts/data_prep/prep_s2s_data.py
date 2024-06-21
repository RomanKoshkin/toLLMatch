from pathlib import Path
import argparse
import pandas as pd
from examples.speech_to_text.data_utils import save_df_to_tsv
import torchaudio


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "tgt_lang"]

def prep_s2s_data(args):
    
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for wav_file in sorted(list(Path(args.path_to_wavs).glob("*.wav"))):
        manifest["id"].append(wav_file.stem)
        manifest["audio"].append(wav_file)
        manifest["n_frames"].append(torchaudio.info(wav_file).num_frames)
        manifest["tgt_text"].append("NA")
        manifest["speaker"].append("NA")
        manifest["tgt_lang"].append("de")
        
    df = pd.DataFrame.from_dict(manifest)
    save_df_to_tsv(df, args.path_to_tsv)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-wavs", "-wav", required=True, type=str)
    parser.add_argument("--path-to-tsv", "-tsv", type=str, required=True)
    args = parser.parse_args()
    prep_s2s_data(args)