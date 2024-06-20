"""
USAGE:
    python make_fleurs_map.py en us
"""

import sys
from dotenv import load_dotenv
import tqdm
import pandas as pd

load_dotenv("../.env", override=True)  # load API keys into
sys.path.append("../")

from datasets import load_dataset


lang = sys.argv[1]
variant = sys.argv[2]
dataset = load_dataset("google/fleurs", f"{lang}_{variant}", streaming=True)

datasetit = iter(dataset["test"])
A = []
for it in tqdm(datasetit):
    A.append(
        {
            "id": it["id"],
            "raw_transcription": it["raw_transcription"],
            "path": it["audio"]["path"].split("/")[-1],
            "lang": lang,
        }
    )
df = pd.DataFrame(A)
df.to_csv(f"../raw_datasets/FLEURS_{lang}.csv", index=False)
