import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utterance_cleaners import (
    covost_utterance_cleaner,
    europarlst_utterance_cleaner,
    mustc_utterance_cleaner,
    general_utterance_cleaner,
)
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

tqdm.pandas(desc="progress")


def remove_noisy_examples(
    df: pd.DataFrame, asr_predictions_file: str, asr_wer_theshold: float
) -> pd.DataFrame:

    # to ensure removal of non-existent ids in both asr and st datasets
    df = df.assign(WER=999)

    # read results
    with open(asr_predictions_file, "r") as file:
        content = [json.loads(line) for line in file]
    ids_to_wer = dict(
        zip([c["id"] for c in content], [float(c["WER"]) for c in content])
    )

    # fill-in values
    df["WER"] = df.progress_apply(lambda x: ids_to_wer.get(x["id"], 999), axis=1)

    df = df.loc[df.WER <= asr_wer_theshold]
    df.drop(columns=["WER"], inplace=True)

    return df


def filter_tsv(args):

    if "MUSTC" in args.tsv_path:
        utterance_cleaner = mustc_utterance_cleaner
    elif "EuroparlST" in args.tsv_path:
        utterance_cleaner = europarlst_utterance_cleaner
    elif "CoVoST" in args.tsv_path:
        utterance_cleaner = covost_utterance_cleaner
    else:
        print("WARNING: Dataset not identified. Using general utterance cleaner")
        utterance_cleaner = general_utterance_cleaner

    st_tsv_path = Path(args.tsv_path)
    st_output_path = Path(args.output_dir_path) / st_tsv_path.name.replace(
        ".tsv", "_filtered.tsv"
    )
    st_output_path.parent.mkdir(parents=True, exist_ok=True)

    df_st = load_df_from_tsv(st_tsv_path)
    print(f"number of ST examples: {len(df_st)}")

    if args.parallel:
        asr_tsv_path = Path(args.tsv_path.replace("_st.tsv", "_asr.tsv"))
        df_asr = load_df_from_tsv(asr_tsv_path)
        print(f"number of parallel ASR examples: {len(df_st)}")
        assert len(df_st) == len(df_asr), "ASR and ST data sizes do not match"

    # target text cleaning
    df_st["tgt_text"] = df_st.progress_apply(
        lambda x: utterance_cleaner(x["tgt_text"]), axis=1
    )
    # removal of empty examples after cleaning
    df_st = df_st.loc[~(df_st.tgt_text == "")]
    print(f"removed empty ST examples, remaining: {len(df_st)}")

    # removal of noisy examples (based on ASR system predictions)
    df_st = remove_noisy_examples(df_st, args.wer_predictions_path, args.wer_threshold)
    print(f"removed noisy examples, remaining: {len(df_st)}")

    if args.parallel:
        df_asr["tgt_text"] = df_asr.progress_apply(
            lambda x: utterance_cleaner(x["tgt_text"]), axis=1
        )
        # removal of empty examples after cleaning
        df_asr = df_asr.loc[~(df_asr.tgt_text == "")]
        print(f"removed empty ASR examples, remaining: {len(df_asr)}")

        asr_idx = set(df_asr.index.tolist())
        st_idx = set(df_st.index.tolist())
        common_idx = asr_idx.intersection(st_idx)
        common_idx = pd.Index(sorted(list(common_idx)))

        df_asr = df_asr.loc[common_idx]
        df_st = df_st.loc[common_idx]

        print("removed not common examples in the ASR and ST datasets")
        print(f"ASR remaining: {len(df_asr)}")
        print(f"ST remaining: {len(df_st)}")

        asr_output_path = Path(args.output_dir_path) / asr_tsv_path.name.replace(
            ".tsv", "_filtered.tsv"
        )
        save_df_to_tsv(df_asr, asr_output_path)
        print(f"Saved filtered ASR at {asr_output_path}")

    save_df_to_tsv(df_st, st_output_path)
    print(f"Saved filtered ST at {st_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv-path",
        "-tsv",
        type=str,
        required=True,
        help="path to the tsv of the ST split",
    )
    parser.add_argument("--wer-predictions-path", "-p", type=str, required=True)
    parser.add_argument("--output-dir-path", "-o", type=str, required=True)
    parser.add_argument(
        "--wer-threshold",
        "-wer",
        type=float,
        default=0.75,
        help="Word-Error-Rate above which an example is considered noisy.",
    )
    parser.add_argument(
        "--parallel",
        "-par",
        action="store_true",
        help="whether to parallel filter the ASR split of the tsv",
    )
    args = parser.parse_args()

    filter_tsv(args)
