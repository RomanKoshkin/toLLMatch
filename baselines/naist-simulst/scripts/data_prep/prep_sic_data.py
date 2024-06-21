#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from fairseq.examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class SIC(Dataset):
    """
    Create a Dataset for NAIST-SIC. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id (MuST-C format)
    """

    lang = "ja"
    SPLITS = ["test", "dev", "train"]

    def __init__(self, root: str, split: str) -> None:
        assert split in self.SPLITS
        _root = Path(root) / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", self.lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in tqdm(groupby(segments, lambda x: x["wav"])):
            wav_path = wav_root / wav_filename
            if not wav_path.is_file():
                print(f"{wav_path} not found ...")
                continue
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                if n_frames == 0:
                    print(f"no speech for {segment['speaker_id']}:{i}. Skipped.")
                    continue
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[self.lang],
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
    root = Path(args.data_root).absolute()
    out_root = Path(args.out_root).absolute()
    out_root.mkdir(exist_ok=True, parents=True)
    if not out_root.is_dir():
        print(f"{out_root.as_posix()} does not exist. Skipped.")

    # Extract features
    audio_root = out_root / ("flac" if args.use_audio_input else "fbank80")
    zip_path = out_root / f"{audio_root.name}.zip"
    generate_zip = not zip_path.is_file()
    if generate_zip:
        audio_root.mkdir(exist_ok=True, parents=True)
        for split in SIC.SPLITS:
            print(f"Fetching split {split}...")

            dataset = SIC(root.as_posix(), split)
            if args.use_audio_input:
                print("Converting audios...")
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
        is_audio=args.use_audio_input,
    )
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SIC.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.append_lang_id:
            manifest["tgt_lang"] = []
        dataset = SIC(args.data_root, split)
        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
            manifest["speaker"].append(speaker_id)
            if args.append_lang_id:
                manifest["tgt_lang"].append("en" if args.task == "asr" else dataset.lang)
        if is_train_split and not args.only_manifest:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(
            df,
            is_train_split=is_train_split,
            min_n_frames=800 if args.use_audio_input else 5,
            max_n_frames=480000 if args.use_audio_input else 3000,
        )
        save_df_to_tsv(df, out_root / f"{split}_{args.task}.tsv")
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
    # Generate vocab
    if not args.only_manifest:
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
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy=None,
            extra={"use_audio_input": True},
        )
    else:
        gen_config_yaml(
            out_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
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
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
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
    parser.add_argument(
        "--only-manifest",
        action="store_true",
        help="only does feature extraction and skips the vocab creation process",
    )
    parser.add_argument("--append-lang-id", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
