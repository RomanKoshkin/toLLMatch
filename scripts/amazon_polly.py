"""
This script demonstrates how to use Amazon Polly to synthesize speech from text.
see documentation in README.md
"""

from dotenv import load_dotenv
from pydub import AudioSegment
from tqdm import tqdm
import sys
import os
import boto3
import pandas as pd

load_dotenv("../.env", override=True)  # load API keys into
sys.path.append("../")


polly_client = boto3.Session(
    aws_access_key_id=os.getenv("AWS_POLLY_KEY"),
    aws_secret_access_key=os.getenv("AWS_POLLY_SECRET_KEY"),
    region_name="ap-southeast-2",
).client("polly")

# ambiguity_big.txt is created using the notebook notebooks/Llama3.ipynb
dd = pd.read_csv("../raw_datasets/ambiguity_big.txt", sep="\t", encoding="utf-16")


for i, row in tqdm(dd.iterrows()):
    response = polly_client.synthesize_speech(
        VoiceId="Joanna",
        OutputFormat="mp3",
        Text=str(row.en),
        TextType="text",
        Engine="neural",
        LanguageCode="en-US",
    )

    with open(f"../raw_datasets/ambiguity_audio_big/a_{i}.mp3", "wb") as f:
        f.write(response["AudioStream"].read())

    # Convert to mono, downsample to 16kHz and export as wav
    audio = AudioSegment.from_mp3(f"../raw_datasets/ambiguity_audio_big/a_{i}.mp3")
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(f"../raw_datasets/ambiguity_audio_big/a_{i}.wav", format="wav", codec="pcm_s16le")
