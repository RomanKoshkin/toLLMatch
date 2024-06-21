from num2words import num2words
import numpy as np
from typing import Tuple
from pathlib import Path
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import torch
from torch.utils.data import DataLoader, Dataset
import jiwer
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
import json
from pathlib import Path
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from examples.speech_to_text.data_utils import load_df_from_tsv
from utterance_cleaners import clean_speaker_name
from typing import List


class AsrDataset(Dataset):
    """
    Dataset object for loading a datapoint, processing it,
    and passing to the dataloader
    """

    def __init__(self, tsv_root: Path, completed_ids: List[str]) -> None:
        super(AsrDataset, self).__init__()

        self.data = load_df_from_tsv(tsv_root)

        self.data = self.data[~self.data.id.isin(set(completed_ids))]

        # sort on n_frames
        self.data = self.data.sort_values("n_frames", axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[str, np.array, str]:

        # get datapoint
        _id, audio_path, tgt_text = self.data.iloc[i][["id", "audio", "tgt_text"]]

        # TODO: remove when fixed
        audio_path = audio_path.replace("MUSTC_v2.0", "MUSTC_v2.0_wav_16k")
        audio_path = audio_path.replace("EuroparlST_wav", "EuroparlST_wav_16k")

        wav_array = get_features_or_waveform(audio_path, need_waveform=True)

        # trasform numbers to words
        try:
            tgt_text = " ".join(
                [
                    num2words(token).replace("and ", "").replace("-", " ")
                    if token.isnumeric()
                    else token
                    for token in tgt_text.split(" ")
                ]
            )
        except:
            print(tgt_text)

        return (_id, wav_array, tgt_text)


def asr_collate_fn(batch: list) -> Tuple[List[str], List[np.array], List[str]]:

    ids = [example[0] for example in batch]
    audio = [example[1] for example in batch]
    tgt_text = [example[2] for example in batch]

    return (ids, audio, tgt_text)


def asr_inference(args) -> None:

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    n_cpu = cpu_count()
    print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

    # specify main device and all devices (if gpu available)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")

    model = Wav2Vec2ForCTC.from_pretrained(args.wav2vec_model_name).eval()
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.wav2vec_model_name)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    print(f"Loaded model and processor: {args.wav2vec_model_name}")

    if len(device_list) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_list, output_device=main_device
        )
    model.to(main_device)

    tsv_path = Path(args.tsv_path)

    # to store results
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    wer_path = output_path / f"{tsv_path.stem}_wer_results.json"

    # check if there are already predictions for some ids
    if wer_path.is_file():
        with open(wer_path, "r") as file:
            content = [json.loads(line) for line in file]
            completed_ids = [c["id"] for c in content]
            wer_list = [c["WER"] for c in content]
    else:
        completed_ids, wer_list = [], []

    # load dataset and initialize dataloader
    dataset = AsrDataset(tsv_path, completed_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size * n_gpu,
        collate_fn=asr_collate_fn,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    print("Loaded dataset and intialized dataloader")

    is_mustc = "MUSTC" in args.tsv_path

    if is_mustc:
        with open(
            "examples/speech_text_joint_to_text/configs/mustc_noise.list", "r"
        ) as f:
            noise_words = f.read().splitlines()
        noise_words = [w.split()[0] for w in noise_words]

    transformations = [
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]

    print(f"Starting inference, results file: {wer_path} is updated every batch")

    # loop through the dataset
    with torch.no_grad():
        for (ids, audio, tgt_text) in tqdm(
            iter(dataloader), miniters=len(dataloader) // 100
        ):

            tokenized_audio = processor(
                audio, return_tensors="pt", padding="longest", sampling_rate=16000
            )
            input_values = tokenized_audio.input_values.to(main_device)
            attention_mask = tokenized_audio.attention_mask.to(main_device)
            logits = model(input_values, attention_mask=attention_mask).logits

            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = [
                trans.lower() for trans in tokenizer.batch_decode(predicted_ids)
            ]

            if is_mustc:
                for w in noise_words:
                    tgt_text = [txt.replace(w, "") for txt in tgt_text]
                tgt_text = [clean_speaker_name(txt) for txt in tgt_text]

            for trnsf in transformations:
                tgt_text = [trnsf(txt) for txt in tgt_text]
                transcriptions = [trnsf(txt) for txt in transcriptions]

            # record WER per datapoint
            wer_results = [
                jiwer.wer(
                    [tgt_text[i].split() if tgt_text[i] != "" else [" "]],
                    [transcriptions[i].split() if transcriptions[i] != "" else [" "]],
                    truth_transform=jiwer.Compose([]),
                    hypothesis_transform=jiwer.Compose([]),
                )
                for i in range(len(ids))
            ]

            # store results for this batch
            with open(wer_path, "a") as f:
                for i in range(len(ids)):
                    json.dump(
                        {
                            "id": ids[i],
                            "WER": wer_results[i],
                            "target text": tgt_text[i],
                            "transcription": transcriptions[i],
                        },
                        f,
                    )
                    f.write("\n")

            # for global scores
            wer_list.extend(wer_results)

    macro_wer = round(sum(wer_list) / len(wer_list), 4)
    print(f"Macro-averaged WER = {macro_wer}")
    with open(output_path / f"{tsv_path.stem}_macro_wer.txt", "w") as f:
        f.write(str(macro_wer))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_path", type=str, required=True, help="path of the tsv file"
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=8,
        help="batch size to be used during inference",
    )
    parser.add_argument(
        "--wav2vec_model_name",
        type=str,
        default="facebook/wav2vec2-large-960h-lv60-self",
    )
    parser.add_argument("--output_path", "-o", type=str, required=True)
    args = parser.parse_args()

    asr_inference(args)
