import argparse
import json

import datasets


def score_translation(
    path_to_refrences: str, path_to_translations: str, path_to_score: str, tgt_lang: str
) -> None:

    # load reference and hypothesis
    with open(path_to_refrences, "r", encoding="utf-8") as f:
        references = f.read().splitlines()
    with open(path_to_translations, "r", encoding="utf-8") as f:
        translations = f.read().splitlines()

    assert len(references) == len(translations)

    references = [[ref] for ref in references]

    sacrebleu = datasets.load_metric("sacrebleu")
    ter = datasets.load_metric("ter")

    results = {}
    if tgt_lang == "de":
        results["sacrebleu"] = round(
            sacrebleu.compute(predictions=translations, references=references)["score"],
            4,
        )
        results["ter"] = round(
            ter.compute(
                predictions=translations, references=references, normalized=True
            )["score"],
            4,
        )
    elif tgt_lang == "ja":
        results["sacrebleu"] = round(
            sacrebleu.compute(
                predictions=translations, references=references, tokenize="ja-mecab"
            )["score"],
            4,
        )
    elif tgt_lang == "zh":
        results["sacrebleu"] = round(
            sacrebleu.compute(
                predictions=translations, references=references, tokenize="zh"
            )["score"],
            4,
        )

    print(results)

    with open(path_to_score, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_refrences", "-ref", required=True, type=str)
    parser.add_argument("--path_to_translations", "-trans", required=True, type=str)
    parser.add_argument("--path_to_score", "-s", type=str, required=True)
    parser.add_argument("--tgt_lang", "-l", type=str, default="de")
    args = parser.parse_args()

    score_translation(
        args.path_to_refrences,
        args.path_to_translations,
        args.path_to_score,
        args.tgt_lang,
    )
