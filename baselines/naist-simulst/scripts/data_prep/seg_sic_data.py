import argparse
import logging
from pathlib import Path
import soundfile as sf
from prep_sic_data import SIC

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args):
    root = Path(args.data_root).absolute()
    lang = "ja"
    split = args.split

#RM    cur_root = root / f"en-{lang}"
#RM    assert cur_root.is_dir(), (
#RM        f"{cur_root.as_posix()} does not exist. Skipped."
#RM    )

    dataset = SIC(root.as_posix(), split)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)
    f_text = open(output / f"{split}.{lang}", "w")
    f_wav_list = open(output / f"{split}.wav_list", "w")
    for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
        sf.write(
            output / f"{utt_id}.wav",
            waveform.squeeze(0).numpy(),
            samplerate=int(sample_rate)
        )
        f_text.write(text + "\n")
        f_wav_list.write(str(output / f"{utt_id}.wav") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=SIC.SPLITS)
    args = parser.parse_args()

    main(args)
