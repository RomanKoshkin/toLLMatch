import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os, sys

load_dotenv("../.env", override=True)
sys.path.append("../")
from openai import OpenAI

SRC_LANG = "en"
try:
    TGT_LANG = TGT_LANG = sys.argv[1]
except Exception as e:
    print("Please provide a target language as a command line argument")
    print(e)
    sys.exit(1)
assert TGT_LANG in ["ru", "es", "fr", "de", "it"], "Invalid target language. Choose from ru, es, fr, de, it"


client = OpenAI(api_key=os.getenv("OPENAI_KEY_MINE"))

# Define the rate limit parameters
RATE_LIMIT = 8  # number of requests per second (OPENAI has a limit of 500 RPM)
CONCURRENT_LIMIT = 100  # number of concurrent requests


async def translate_sentence(session, src_lang, tgt_lang, sentence):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_KEY_MINE')}", "Content-Type": "application/json"}
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. You are great at producing accurate and elegant translation.",
            },
            {
                "role": "user",
                "content": f"Translate the following {src_lang} text into {tgt_lang}. Just give me the translation and nothing else (no comments, no explanations, please).\n Text:\n{sentence}",
            },
        ],
        "model": "gpt-3.5-turbo",
    }
    async with session.post(url, headers=headers, json=data) as response:
        if response.status == 200:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(response.status)
            print(dir(response.status))
            print(f"Failed to translate {sentence} with status {response.status}")
            return None


async def translate_sentences(df, src_lang, tgt_lang):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
        tasks = []
        progress_bar = tqdm(total=len(df))

        def update_progress_bar(task):
            progress_bar.update(1)

        async def rate_limited_translate(i, source_text):
            async with semaphore:
                await asyncio.sleep(i / RATE_LIMIT)
                return i, await translate_sentence(session, src_lang, tgt_lang, source_text)

        for i, source_text in enumerate(df.raw_transcription):
            task = asyncio.create_task(rate_limited_translate(i, source_text))
            task.add_done_callback(update_progress_bar)  # Update the progress bar for each completed task
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(result)
            else:
                i, translation = result
                df.at[i, f"{TGT_LANG}"] = translation

        progress_bar.close()  # Ensure the progress bar is closed after all tasks are completed


def main():
    df = pd.read_csv("../raw_datasets/FLEURS_en.csv")
    asyncio.run(translate_sentences(df, SRC_LANG, TGT_LANG))
    df.to_csv(f"../raw_datasets/FLEURS_en_{TGT_LANG}.csv", index=False)


if __name__ == "__main__":
    main()
