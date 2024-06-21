# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace
from omegaconf import II, MISSING
from dataclasses import dataclass, field
from typing import Optional

from fairseq import utils, scoring
from fairseq.logging import metrics
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform,
)
from fairseq.data.audio.speech_augmentation_dataset import SpeechAugmentationDataset
from fairseq.data.audio.speech_distillation_dataset import SpeechDistillationDataset

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.scoring.wer import WerScorerConfig
from fairseq.scoring.bleu import SacrebleuConfig
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import safe_hasattr


logger = logging.getLogger(__name__)

@dataclass
class DataAugmentationConfig(FairseqDataclass):
    p_augm: float = field(
        default=0,
        metadata={"help":
            "The probability that data augmentation is applied to an example."
            "0 means that data augmentation is not active."
        }
    )
    tempo: str = field(
        default="1,1",
        metadata={"help": "The range from which to sample the tempo factor"}
    )
    pitch: str = field(
        default="0,0",
        metadata={"help":
            "The range from which to sample the pitch value."
            "Measured in cents (i.e. 100ths of a semitone)"
        }
    )
    echo_delay: str = field(
        default="0,0",
        metadata={"help":
            "The range from which to sample the echo delay value."
            "Measured in milliseconds"
        }
    )
    echo_decay: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the echo decay factor."}
    )

@dataclass
class KnowledgeDistillationConfig(FairseqDataclass):
    path: str = field(
        default="",
        metadata={"help": "path to teacher outputs: ${path}/${split}/*.pt"}
    )

@dataclass
class SpeechToTextTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "manifest root path"}
    )
    data_config_yaml: str = field(
        default="config.yaml",
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    max_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )

    # Reporting metrics during training
    eval_wer: bool = field(
        default=False,
        metadata={"help": "compute WER on the validation set"}
    )
    eval_wer_config: WerScorerConfig = field(
        default_factory=lambda: WerScorerConfig("wer"),
        metadata={"help": "WER scoring configuration"},
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "compute SacreBLEU on the validation set"}
    )
    eval_bleu_config: SacrebleuConfig = field(
        default_factory=lambda: SacrebleuConfig("sacrebleu"),
        metadata={"help": "SacreBLEU scoring configuration"},
    )
    eval_gen_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "generaton config for evaluating during training"},
    )

    # Inherit from other configs
    train_subset: str = II("dataset.train_subset")
    seed: int = II("common.seed")
    
    data_augmentation: Optional[DataAugmentationConfig] = field(
        default=None,
        metadata={"help": "Data augmentation arguments"},
    )
    knowledge_distillation: Optional[KnowledgeDistillationConfig] = field(
        default=None,
        metadata={"help": "Knowledge distillation arguments"}
    )
    sampling_ratios: str = field(
        default="1",
        metadata={"help": "sampling ratios of the train subsets"}
    )
    interactive_tgt_lang: Optional[str] = field(
        default=None,
        metadata={"help": "Target language to be used with Fairseq's interactive mode."}
    )


@register_task("speech_to_text", dataclass=SpeechToTextTaskConfig)
class SpeechToTextTask(FairseqTask):

    def __init__(self, cfg, tgt_dict):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(cfg.data) / cfg.data_config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        self.pre_tokenizer = self.build_tokenizer(cfg)
        self.bpe_tokenizer = self.build_bpe(cfg)
        self.scorers = []
        if self.cfg.eval_wer:
            self.scorers.append(
                scoring.build_scorer(cfg.eval_wer_config, self.tgt_dict)
            )
        if self.cfg.eval_bleu:
            self.scorers.append(
                scoring.build_scorer(cfg.eval_bleu_config, self.tgt_dict)
            )
            
        # effect parameters for data augmentation
        self.effects_info = {
            "tempo": list(map(float, cfg.data_augmentation.tempo.split(","))),
            "pitch": list(map(int, cfg.data_augmentation.pitch.split(","))),
            "echo": {
                "delay": list(map(int, cfg.data_augmentation.echo_delay.split(","))),
                "decay": list(map(float, cfg.data_augmentation.echo_decay.split(",")))
            }
        } if cfg.data_augmentation is not None and cfg.data_augmentation.p_augm > 0 else None
        self.sampling_ratios = list(map(float, cfg.sampling_ratios.split(",")))
        
        self.use_kd = cfg.knowledge_distillation is not None and cfg.knowledge_distillation != ""

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.cfg.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        data_cfg = S2TDataConfig(Path(cfg.data) / cfg.data_config_yaml)
        dict_path = Path(data_cfg.vocab_filename)
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        if cfg.eval_wer:
            if cfg.eval_wer_config.wer_tokenizer == "none":
                logger.warning(
                    "You are not using any tokenizer for WER scoring. Using '13a' is recommended."
                )
            if not cfg.eval_wer_config.wer_lowercase:
                logger.warning(
                    "You are not lowercasing before WER scoring."
                )
            if not cfg.eval_wer_config.wer_remove_punct:
                logger.warning(
                    "You are not removing punctuation before WER scoring."
                )

        return cls(cfg, tgt_dict)

    def build_criterion(self, cfg):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and cfg.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "criterion.ignore_prefix_size=1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(cfg, self)

    def load_dataset(self, split, epoch=1, **kwargs):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.pre_tokenizer,
            self.bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
            speaker_to_id=self.speaker_to_id,
            sampling_ratios=self.sampling_ratios,
            no_standardize_audio=is_train_split and self.effects_info is not None
        )
        
        if is_train_split:
            if self.use_kd:
                self.datasets[split] = SpeechDistillationDataset(
                    self.datasets[split],
                    self.cfg.knowledge_distillation.path,
                    pad_idx=self.tgt_dict.pad()
                )
            
            if self.effects_info is not None:
                logger.info(f"Using data augmentation on '{split}' with probability "
                            f"of {self.cfg.data_augmentation.p_augm} and the following "
                            f"configuration: {self.effects_info}")
                self.datasets[split] = SpeechAugmentationDataset(
                    self.datasets[split],
                    effects_info=self.effects_info,
                    p_augm=self.cfg.data_augmentation.p_augm,
                    max_src_len=self.cfg.max_source_positions
                )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, cfg, from_checkpoint=False):
        model = super(SpeechToTextTask, self).build_model(cfg, from_checkpoint)
        self.sequence_generator = self.build_generator(
            [model],
            self.cfg.eval_gen_config
        )
        # Trick: update model configuration globally
        if not safe_hasattr(cfg.encoder, 'pre_args'):
            cfg.encoder.pre_args = model.encoder.cfg_.pre_args
        if not safe_hasattr(cfg.decoder, 'pre_args'):
            cfg.decoder.pre_args = model.decoder.cfg_.pre_args

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        def decode(toks):
            s = self.tgt_dict.string(toks)
            if self.bpe_tokenizer:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer:
                s = self.pre_tokenizer.decode(s)
            return s

        if len(self.scorers) > 0:
            prefix_tokens = sample['target'][:, 0].unsqueeze(1) if self.data_cfg.prepend_tgt_lang_tag else None
            gen_out = self.inference_step(self.sequence_generator, [model], sample, prefix_tokens=prefix_tokens)
            for i in range(len(gen_out)):
                ref_tok = utils.strip_pad(sample["target"][i], self.tgt_dict.pad()).int().cpu()
                pred_tok = gen_out[i][0]["tokens"].int().cpu()
                if self.data_cfg.prepend_tgt_lang_tag:
                    ref_tok = ref_tok[1:]
                    pred_tok = pred_tok[1:]
                ref = decode(ref_tok)
                pred = decode(pred_tok)
                for s in self.scorers:
                    s.add_string(ref, pred)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        for s in self.scorers:
            # Log score just for validation set
            if (s.cfg._name == 'wer' and s.ref_length > 0) or \
                (s.cfg._name == 'sacrebleu' and len(s.pred) > 0):
                metrics.log_scalar(s.cfg._name, round(s.score(), 2))
                if safe_hasattr(s, "reset"):
                    s.reset()
                else:
                    s.ref = []
                    s.pred = []

    def build_generator(
        self,
        models,
        cfg,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and cfg.prefix_size < 1:
            raise ValueError(
                'Please set "generation.prefix_size>=1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        return super().build_generator(
            models, cfg, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, cfg):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, cfg):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
