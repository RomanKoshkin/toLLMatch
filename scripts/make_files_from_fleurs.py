import sys
import os
import re
import pandas as pd


PATH_TO_SOURCES = "../evaluation/SOURCES"
PATH_TO_TARGETS = "../evaluation/OFFLINE_TARGETS"
PATH_TO_RAW_DATASETS = "../raw_datasets"

src_l = sys.argv[1]
src_variant = sys.argv[2]
tgt_l = sys.argv[3]

df = pd.read_csv(f"../raw_datasets/FLEURS_{src_l}_{tgt_l}.csv")

try:
    os.remove(f"{PATH_TO_SOURCES}/fleurs_{src_l}_{tgt_l}")
except FileNotFoundError:
    pass
try:
    os.remove(f"{PATH_TO_SOURCES}/fleurs_{src_l}_{tgt_l}.txt")
except FileNotFoundError:
    pass
try:
    os.remove(f"{PATH_TO_TARGETS}/fleurs_{src_l}_{tgt_l}.txt")
except FileNotFoundError:
    pass

for i, r in df.iterrows():
    with open(f"{PATH_TO_SOURCES}/fleurs_{src_l}_{tgt_l}", "a") as f:
        f.write(f"{PATH_TO_RAW_DATASETS}/fleurs_audio_{src_l}_{src_variant}/{r.path}\n")
    with open(f"{PATH_TO_SOURCES}/fleurs_{src_l}_{tgt_l}.txt", "a") as f:
        text = re.sub(r"\n", "", r.raw_transcription)
        f.write(f"{text}\n")
    with open(f"{PATH_TO_TARGETS}/fleurs_{src_l}_{tgt_l}.txt", "a") as f:
        text = re.sub(r"\n", "", r[tgt_l])
        f.write(f"{text}\n")
