import ast
from argparse import Namespace
from tqdm import tqdm

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    return _main(cfg)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def calc_longest_common_prefix(partial_tokens: torch.Tensor, full_tokens: torch.Tensor) -> torch.Tensor:
    """calculate longest common prefix between partial tokens and full-sentence tokens

    Args:
        partial_tokens (torch.Tensor): decoding result using partial source token indices, shape of (L)
        full_tokens (torch.Tensor): decoding result using full source token indices, shape of (L)

    Returns:
        torch.Tensor: token indices of longest common prefix
    """
    longest_common_prefix = []
    
    for partial_token, full_token in zip(
        partial_tokens,
        full_tokens,
    ):
        if partial_token == full_token:
            longest_common_prefix.append(partial_token.item())
        else:
            break
    
    return torch.Tensor(longest_common_prefix).to(torch.int64)


def is_boundary(current_prefix_tokens: torch.Tensor, longest_common_prefix: torch.Tensor) -> bool:
    """detect the boundary for prefix alignment using current prefix tokens and longest common prefix 

    Args:
        current_prefix_tokens (torch.Tensor): prefix tokens used for decoding, shape of (N, L)
        longest_common_prefix (torch.Tensor): longest common prefix calculated after decoding, shape of (L)

    Returns:
        bool: if it's boundary, returns True
    """
    
    current_prefix_length = current_prefix_tokens.shape[1]
    longest_common_prefix_length = longest_common_prefix.shape[0]
    
    return current_prefix_length < longest_common_prefix_length


def _main(cfg: DictConfig):
    utils.import_user_module(cfg.common)
    
    output_file = open(cfg.task.output_filepath, mode="w", encoding="utf-8")

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    generator = task.build_generator(
        models, cfg.generation,
    )

    dataset = task.dataset(cfg.dataset.gen_subset)
    
    # Setup file header
    print(
        "id\taudio\tn_frames\ttgt_text\tspeaker\ttgt_lang",
        file=output_file,
    )
    
    estimated_steps = 0
    for n_frames in dataset.n_frames:
        # full-translation step
        estimated_steps += 1
        # part-translation step
        estimated_steps += n_frames // cfg.task.pre_decision_size
    
    bar = tqdm(total=estimated_steps)
    
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample_id = sample["id"][0]

        # lang_id in the `prefix_tokens`
        # shape of (N, T)
        lang_id = sample["target"][:, :1]
        
        full_trans = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=lang_id,
            constraints=None,
        )[0][:cfg.generation.nbest][0]
        
        full_trans_tokens, full_trans_str, _ = utils.post_process_prediction(
            hypo_tokens=full_trans["tokens"].int().cpu(),
            src_str="",
            alignment=full_trans["alignment"],
            align_dict=None,
            tgt_dict=tgt_dict,
            remove_bpe=cfg.common_eval.post_process,
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
        )
        bar.update(1)
        
        src_tokens = sample["net_input"]["src_tokens"] # shape of (N, T)
        src_lengths = sample["net_input"]["src_lengths"] # shape of (N)
        prefix_tokens = lang_id # shape of (N, T)
        
        longest_length = 0
        full_translation_length = full_trans_tokens.shape[0]
        
        for duration in range(
            cfg.task.pre_decision_size,
            int(src_lengths[0]) + 1,
            cfg.task.pre_decision_size,
        ):  
            # create source prefix sample
            partial_sample = {
                "net_input": {
                    "src_tokens": src_tokens[:, :duration],
                    "src_lengths": torch.Tensor([duration]).to(torch.int64).cuda()
                }
            }
            
            partial_trans = task.inference_step(
                generator,
                models,
                partial_sample,
                prefix_tokens=prefix_tokens
            )[0][:cfg.generation.nbest][0]
            
            # shape of (L)
            partial_trans_tokens, _, _ = utils.post_process_prediction(
                hypo_tokens=partial_trans["tokens"].int().cpu(),
                src_str="",
                alignment=partial_trans["alignment"],
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            bar.update(1)
            
            longest_common_prefix = calc_longest_common_prefix(
                partial_trans_tokens, full_trans_tokens
            )
            
            if is_boundary(prefix_tokens, longest_common_prefix):
                # update the prefix tokens for forced decoding
                prefix_tokens = longest_common_prefix.unsqueeze(0).cuda()
                
                # convert tensor to str and remove lang_id
                longest_common_prefix_str = task.target_dictionary.string(
                    longest_common_prefix[1:] # remove lang_id
                )

                # output prefix translation pair
                print(
                    f"{dataset.ids[sample_id]}\t"\
                    f"{dataset.audio_paths[sample_id]}\t"\
                    f"{duration}\t"\
                    f"{longest_common_prefix_str}\t"\
                    f"{dataset.speakers[sample_id]}\t"\
                    f"{dataset.tgt_langs[sample_id]}",
                    file=output_file,
                )
                
                # update longest length
                longest_length = prefix_tokens.shape[1]
        
        # if prefix tranlation length reaches full translation length, 
        # output full-translation pair
        if longest_length < full_translation_length:
            print(
                f"{dataset.ids[sample_id]}\t"\
                f"{dataset.audio_paths[sample_id]}\t"\
                f"{dataset.n_frames[sample_id]}\t"\
                f"{full_trans_str}\t"\
                f"{dataset.speakers[sample_id]}\t"\
                f"{dataset.tgt_langs[sample_id]}",
                file=output_file,
            )


if __name__ == "__main__":
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
