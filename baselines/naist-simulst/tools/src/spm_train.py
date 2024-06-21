"""
Train SentencePiece model.
"""
import os
import argparse
import codecs
import subprocess
import sentencepiece as spm

parser = argparse.ArgumentParser(description="sentencepiece_trainer.py")
parser.add_argument("--input", nargs="*", required=True, 
                    help="input files to train sentencepiece model")
parser.add_argument("--model_prefix", required=True, 
                    help="output model prefix")
parser.add_argument("--model_type", default="unigram",
                    choices=["unigram", "bpe", "word", "char"],
                    help="model algorithm")
parser.add_argument("--vocab_size", type=int, required=True, 
                    help="vocabulary size")
parser.add_argument("--pad_id", type=int, default=-1, 
                    help="Override PAD (<pad>) id. Set -1 to disable PAD.")
parser.add_argument("--input_sentence_size", type=int, default=0,
                    help="sample <size> sentences")
parser.add_argument("--shuffle_input_sentence", type=bool, default=False,
                    help="Randomly sample input sentences in advance. \
                          Valid when --input_sentence_size > 0")
args = parser.parse_args()


def main():
    cmd = "cat"
    for s in args.input:
        assert os.path.exists(s)
        cmd += f" {s}"
    cmd += f" > {args.model_prefix}.data"
    subprocess.Popen(cmd, shell=True).wait()

    spm.SentencePieceTrainer.Train(
        f"--input={args.model_prefix}.data \
          --model_prefix={args.model_prefix} \
          --model_type={args.model_type} \
          --vocab_size={args.vocab_size} \
          --pad_id={args.pad_id} \
          --input_sentence_size={args.input_sentence_size} \
          --shuffle_input_sentence=true"
    )


if __name__ == "__main__":
    main()
