import torchaudio
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import pandas as pd

tqdm.pandas(desc="progress")

TGT_LANGUAGES = ["de", "zh", "ja"]
SPLITS = ["train", "dev", "test"]


def prepare_covost_tsv(args):

    assert args.tgt_lang in TGT_LANGUAGES
    root = Path(args.data_root)
    out_root = Path(args.out_root).absolute()

    lang_pair = f"{args.src_lang}-{args.tgt_lang}"

    tgt_lang_ = f"{args.tgt_lang}-CN" if args.tgt_lang == "zh" else args.tgt_lang
    df = load_df_from_tsv(
        root
        / args.src_lang
        / args.covost_name
        / lang_pair
        / f"covost_v2.{args.src_lang}_{tgt_lang_}.tsv"
    )

    if args.task == "asr":
        df_asr = pd.concat(
            [
                load_df_from_tsv(root / args.src_lang / f"{split}.tsv")
                for split in SPLITS + ["other", "validated", "invalidated"]
            ],
            ignore_index=True,
        )
        df_asr.set_index("path", inplace=True, drop=True)

    for split in SPLITS:
        print(f"Preparing {lang_pair} - {split} - {args.task}")

        lang_pair_path = out_root / args.src_lang / args.covost_name / lang_pair
        st_split_path = lang_pair_path / f"{split}_st.tsv"
        asr_split_path = lang_pair_path / f"{split}_asr.tsv"


        if args.task == "asr" and st_split_path.is_file():
            print("Loading tsv for ST ... ")
            df_split = load_df_from_tsv(st_split_path)
            df_split["path"] = df_split.progress_apply(
                lambda x: x["audio"].split("/")[-1], axis=1
            )

        else:
            df_split = df.loc[df.split == split, :].copy()
            df.drop(df_split.index, inplace=True)

            df_split["id"] = df_split.progress_apply(
                lambda x: x["path"].split(".")[0], axis=1
            )
            df_split["audio"] = df_split.progress_apply(
                lambda x: str(
                    root
                    / args.src_lang
                    / args.clips_name
                    / x["path"]
                ),
                axis=1,
            )
            df_split["n_frames"] = df_split.progress_apply(
                lambda x: torchaudio.info(x["audio"]).num_frames
                if Path(x["audio"]).is_file()
                else 0,
                axis=1,
            )
            df_split.rename(columns={"translation": "tgt_text"}, inplace=True)
            df_split["speaker"] = "NA"

            prev_size = len(df_split)
            df_split = df_split.loc[df_split.n_frames != 0, :]
            print(
                f"{prev_size - len(df_split)} audio files were not found and were removed"
            )

            if split == "train":
                prev_size = len(df_split)
                df_split = df_split.loc[df_split.n_frames >= 800, :]
                df_split = df_split.loc[df_split.n_frames <= 480000, :]
                print(
                    f"{prev_size - len(df_split)} examples were removed due to small/large num_frames"
                )

            prev_size = len(df_split)
            df_split = df_split.loc[df_split.tgt_text != "[TO REMOVE]", :]
            df_split = df_split.loc[df_split.tgt_text != "", :]
            print(
                f"{prev_size - len(df_split)} examples were removed due to empty target text"
            )

        if args.task == "asr":
            df_split["tgt_text"] = df_asr.loc[df_split.path, "sentence"].drop_duplicates().tolist()

        df_split = df_split[["id", "audio", "n_frames", "tgt_text", "speaker"]]
        
        if args.append_lang_id:
            df_split["tgt_lang"] = "en" if args.task=="asr" else args.tgt_lang
        
        save_df_to_tsv(
            df_split,
            st_split_path if args.task == "st" else asr_split_path,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        "-d",
        required=True,
        type=str,
        help="data root with sub-folders for each language <root>/<src_lang>",
    )
    parser.add_argument("--out-root", required=True, type=str)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", required=True, type=str)
    parser.add_argument("--task", type=str, choices=["asr", "st"], default="st")
    parser.add_argument("--clips-name", type=str, default="clips_16k_mono")
    parser.add_argument("--covost-name", type=str, default="CoVoST")
    parser.add_argument("--append-lang-id", action="store_true")
    args = parser.parse_args()

    prepare_covost_tsv(args)
