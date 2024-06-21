"""
Tokenize 1 line 1 sentence text using pre-trained SentencePiece model.
"""
import argparse
import codecs
import sentencepiece as spm
from pathlib import Path

parser = argparse.ArgumentParser(description="spm_encode.py")
parser.add_argument("--input", required=True,
                    help="input file to filter/encode")
parser.add_argument("--output", required=True,
                    help="path to save encoder output")
parser.add_argument("--model", required=True,
                    help="sentencepiece model to use for encoding")
args = parser.parse_args()

def main():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.model)
    in_txt=Path(args.input)
    out_txt=Path(args.output)
    in_lines = []
    out_lines = []
    with in_txt.open("r", encoding="utf-8") as f:
        for line in f.readlines():
            in_lines.append(line)
        
    for in_line in in_lines:
        pieces = tokenizer.EncodeAsPieces(in_line)
        out_lines.append(" ".join(pieces).strip()+"\n")
    
    with out_txt.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)
    
    print(f"{args.input}: {len(in_lines)}")
    print(f"{args.output}: {len(out_lines)}")

if __name__ == "__main__":
    main()
