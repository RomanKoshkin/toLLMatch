import argparse
import logging
import shutil
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from examples.speech_to_text.data_utils import (
    cal_gcmvn_stats,
    convert_waveform,
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from fairseq.data.audio.audio_utils import get_waveform
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class EuroparlST(Dataset):
    """
    Create a Dataset for EuroparlST.
    Each item is a tuple of the form: waveform, sample_rate, source utterance,
    target utterance, speaker_id, target language, utterance_id
    """

    SPLITS = ["train", "dev", "test"]
    LANGPAIRS = [
        f"{l1}-{l2}"
        for l2 in ["en", "fr", "de", "it", "es", "pt", "pl", "ro"]
        for l1 in ["en", "fr", "de", "it", "es", "pt", "pl", "ro"]
        if l1 != l2
    ]

    def __init__(self, src_lang_root: Path, lang_pair: str, split: str) -> None:
        assert split in self.SPLITS and lang_pair in self.LANGPAIRS
        src_lang, tgt_lang = lang_pair.split("-")
        wav_root = src_lang_root / "audios"
        txt_root = src_lang_root / tgt_lang / split
        assert src_lang_root.is_dir() and wav_root.is_dir() and txt_root.is_dir()

        # Create speaker dictionary
        with open(txt_root / "speeches.lst") as f:
            speeches = [r.strip() for r in f]
        with open(txt_root / "speakers.lst") as f:
            speakers = [r.strip() for r in f]
        assert len(speeches) == len(speakers)
        spk_dict = {spe: spk.split("_")[-1] for spe, spk in zip(speeches, speakers)}

        # Generate segments dictionary
        with open(txt_root / "segments.lst") as f:
            segments = [
                {
                    "wav": r.split(" ")[0],
                    "offset": float(r.split(" ")[1]),
                    "duration": float(r.split(" ")[2].strip()) - float(r.split(" ")[1]),
                    "speaker_id": spk_dict[r.split(" ")[0]],
                }
                for r in f
            ]

        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"segments.{_lang}") as f:
                utts = [r.strip() for r in f]
            assert len(segments) == len(utts)
            for i, u in enumerate(utts):
                segments[i][_lang] = u

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / (wav_filename + ".wav")
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    src_lang, tgt_lang = args.lang_pair.split("-")
    src_lang_root = Path(args.data_root).absolute() / src_lang
    out_root = Path(args.out_root).absolute() / src_lang

    # Extract features
    audio_root = out_root / ("flac" if args.use_audio_input else "fbank80")
    zip_path = out_root / f"{audio_root.name}.zip"
    generate_zip = not zip_path.is_file()
    if generate_zip:
        audio_root.mkdir(exist_ok=True, parents=True)
        for split in EuroparlST.SPLITS:
            print(f"Fetching split {split}...")
            dataset = EuroparlST(src_lang_root, args.lang_pair, split)

            if args.use_audio_input:
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    tgt_sample_rate = 16_000
                    _wavform, _ = convert_waveform(
                        waveform,
                        sample_rate,
                        to_mono=True,
                        to_sample_rate=tgt_sample_rate,
                    )
                    sf.write(
                        (audio_root / f"{utt_id}.flac").as_posix(),
                        _wavform.T.numpy(),
                        tgt_sample_rate,
                    )
            else:
                print("Extracting log mel filter bank features...")
                gcmvn_feature_list = []
                if split == "train" and args.cmvn_type == "global":
                    print("And estimating cepstral mean and variance stats...")

                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    features = extract_fbank_features(
                        waveform, sample_rate, audio_root / f"{utt_id}.npy"
                    )
                    if split == "train" and args.cmvn_type == "global":
                        if len(gcmvn_feature_list) < args.gcmvn_max_num:
                            gcmvn_feature_list.append(features)

                if split == "train" and args.cmvn_type == "global":
                    # Estimate and save cmv
                    stats = cal_gcmvn_stats(gcmvn_feature_list)
                    with open(out_root / "gcmvn.npz", "wb") as f:
                        np.savez(f, mean=stats["mean"], std=stats["std"])
        # Pack features into ZIP
        print("ZIPing audios/features...")
        create_zip(audio_root, zip_path)

    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(
        zip_path,
        is_audio=args.use_audio_input
    )

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in EuroparlST.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.append_lang_id:
            manifest["tgt_lang"] = []
        dataset = EuroparlST(src_lang_root, args.lang_pair, split)
        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
            manifest["speaker"].append(speaker_id)
            if args.append_lang_id:
                manifest["tgt_lang"].append(src_lang if args.task == "asr" else tgt_lang)
        if is_train_split and not args.only_manifest:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(
            df,
            is_train_split=is_train_split,
            min_n_frames=800 if args.use_audio_input else 5,
            max_n_frames=480000 if args.use_audio_input else 3000,
        )
        file_name =f"{args.lang_pair}_{split}_{args.task}.tsv"
        save_df_to_tsv(df, out_root / file_name)

    # Generate vocab
    if not args.only_manifest:
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"{args.lang_pair}_spm_{args.vocab_type}{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                out_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        if args.use_audio_input:
            gen_config_yaml(
                out_root,
                spm_filename=spm_filename_prefix + ".model",
                yaml_filename=f"config_{args.task}.yaml"
                if args.task == "asr"
                else f"{args.lang_pair}_config_{args.task}.yaml",
                specaugment_policy=None,
                extra={"use_audio_input": True},
            )
        else:
            gen_config_yaml(
                out_root,
                spm_filename=spm_filename_prefix + ".model",
                yaml_filename=f"config_{args.task}.yaml"
                if args.task == "asr"
                else f"{args.lang_pair}_config_{args.task}.yaml",
                specaugment_policy="lb",
                cmvn_type=args.cmvn_type,
                gcmvn_path=(
                    out_root / "gcmvn.npz" if args.cmvn_type == "global" else None
                ),
            )

    # Clean up
    if generate_zip:
        shutil.rmtree(audio_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        "-d",
        required=True,
        type=str,
        help="data src_lang_root with sub-folders for each language <src_lang_root>/<src_lang>",
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--lang-pair", required=True, type=str)
    parser.add_argument(
        "--cmvn-type",
        default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization",
    )
    parser.add_argument(
        "--gcmvn-max-num",
        default=150000,
        type=int,
        help="Maximum number of sentences to use to estimate global mean and "
        "variance",
    )
    parser.add_argument("--out-root", required=True, type=str)
    parser.add_argument("--use-audio-input", action="store_true")
    parser.add_argument("--only-manifest", action="store_true")
    parser.add_argument("--append-lang-id", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
