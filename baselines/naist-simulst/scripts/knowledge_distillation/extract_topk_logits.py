import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from source_dataset import SourceDataset

tqdm.pandas(desc="progress")


def prepare_corpus_for_kd(args):

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    print(f"Number of cuda devices: {n_gpu}")

    # specify main device and all devices (if gpu available)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")
    model = MBartForConditionalGeneration.from_pretrained(args.model_name).eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(
        args.model_name, src_lang="en_XX", tgt_lang=args.tgt_lang
    )

    if len(device_list) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_list, output_device=main_device
        )
    model.to(main_device)

    path_to_output = Path(args.path_to_output)
    if path_to_output.exists():
        completed_ids = [file.stem for file in path_to_output.glob("*.pt")]
    else:
        completed_ids = []
        path_to_output.mkdir(parents=True, exist_ok=True)

    print(f"Completed ids {len(completed_ids)}")

    dataset = SourceDataset(
        args.path_to_asr_tsv, args.path_to_st_tsv, tokenizer, completed_ids
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size * n_gpu,
        collate_fn=dataset.collater,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    with torch.no_grad():
        for sgm_ids, source_texts, target_texts in tqdm(iter(dataloader)):

            inputs = tokenizer(source_texts, return_tensors="pt", padding="longest")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(target_texts, return_tensors="pt", padding="longest")

            tgt_seq_lens = labels.attention_mask.sum(dim=1)

            inputs = inputs.to(main_device)
            labels = labels.to(main_device)

            logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels.input_ids,
            )["logits"]

            topk_outputs, topk_indices = torch.topk(logits, k=args.top_k, dim=-1)
            bs = len(topk_outputs)

            for i in range(bs):
                seq_len_i = tgt_seq_lens[i]
                topk_outputs_i = topk_outputs[i, args.remove_prefix_len:seq_len_i, :].detach().cpu()
                topk_indices_i = topk_indices[i, args.remove_prefix_len:seq_len_i, :].detach().cpu()

                torch.save(
                    {
                        "topk_indices": topk_indices_i,
                        "topk_outputs": topk_outputs_i,
                    },
                    path_to_output / f"{sgm_ids[i]}.pt",
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-asr-tsv",
        "-asr",
        required=True,
        type=str,
        help="path to the asr tsv with the english transcription under tgt_text",
    )
    parser.add_argument(
        "--path-to-st-tsv",
        "-st",
        required=True,
        type=str,
        help="path to the asr tsv with the english transcription under tgt_text",
    )
    parser.add_argument("--path-to-output", "-o", type=str, required=True)
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="facebook/mbart-large-50-one-to-many-mmt",
    )
    parser.add_argument("--tgt-lang", "-t", type=str, default="de_DE")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=8,
        help="keep the top-k most probable tokens",
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    parser.add_argument("--remove-prefix-len", "-pref", type=int, default=1)
    args = parser.parse_args()

    prepare_corpus_for_kd(args)
