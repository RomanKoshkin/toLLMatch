"""
Detokenize 1 line 1 sentence text using pre-trained SentencePiece model.
"""
import argparse
import codecs
import sentencepiece as spm

parser = argparse.ArgumentParser(description="spm_decode.py")
parser.add_argument("-i", "--input", required=True,
                    help="input file to decode")
parser.add_argument("-o", "--output", required=True,
                    help="path to save decoder output")
parser.add_argument("--model", required=True,
                    help="sentencepiece model to use for decoding")
args = parser.parse_args()


def main():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.model)
    in_file = codecs.open(args.input, "r", "utf-8")
    out_file = codecs.open(args.output, "w", "utf-8")
    while True:
        sline = in_file.readline() 
        if sline == "":
            break
        detok = tokenizer.decode(sline.split())
        out_file.write(detok+"\n")
    in_file.close()
    out_file.close()


if __name__ == "__main__":
    main()
