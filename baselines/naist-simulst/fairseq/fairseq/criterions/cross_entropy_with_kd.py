# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# TODO give credits to
# https://github.com/mgaido91/FBK-fairseq-ST/blob/master/fairseq/criterions/knowledge_distillation.py

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils, metrics
from omegaconf import II
from . import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class CrossEntropyWithKDCriterionConfig(FairseqDataclass):
    teacher_lambda: float = field(
        default=0,
        metadata={"help": "The weight of the KD criterion. 0 means that KD is not active"}
    )
    teacher_temperature: float = field(
        default=1,
        metadata={"help": "Temperature to normalize the distributions of student and teacher."}
    )
    teacher_ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    force_calculate_nll: bool = field(
        default=False,
        metadata={"help": "To calculate nll loss even though lambda == 1"}
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    

@register_criterion("cross_entropy_with_kd", dataclass=CrossEntropyWithKDCriterionConfig)
class CrossEntropyWithKDCriterion(FairseqCriterion):
    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg

    @classmethod
    def build_criterion(cls, cfg: CrossEntropyWithKDCriterionConfig, task):
        assert cfg.teacher_lambda > 0, \
            "Warning: teacher lambda is zero." \
            "It's better to use label-smoothed cross-entropy instead"
        assert cfg.teacher_lambda <= 1, \
            "Warning: teacher weight cannot be larger than 1"           
        return cls(cfg, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        
        # KD from the teacher
        if "teacher_output" in sample.keys():
            net_output_scaled = (net_output[0] / self.cfg.teacher_temperature, net_output[1])
            lprobs_scaled = model.get_normalized_probs(net_output_scaled, log_probs=True)
            
            if self.cfg.ignore_prefix_size > 0:
                lprobs_scaled = lprobs_scaled[:, self.cfg.ignore_prefix_size:, :].contiguous()
            lprobs_scaled = lprobs_scaled.view(-1, lprobs_scaled.size(-1))
            
            teacher_idxs = sample["teacher_output"]["topk_indices"][:, self.cfg.teacher_ignore_prefix_size:]
            teacher_outs = sample["teacher_output"]["topk_outputs"][:, self.cfg.teacher_ignore_prefix_size:]

            teacher_probs = F.softmax(teacher_outs / self.cfg.teacher_temperature, dim=-1)
            teacher_idxs = teacher_idxs.view(-1, teacher_idxs.shape[-1])
            teacher_probs = teacher_probs.view(-1, teacher_probs.shape[-1])

            lprobs_scaled = lprobs_scaled.gather(dim=-1, index=teacher_idxs.long()).contiguous()
            teacher_loss = -lprobs_scaled * teacher_probs

            # Ignore paddings
            teacher_mask = teacher_idxs.eq(self.padding_idx)
            teacher_loss.masked_fill_(teacher_mask, 0.0)
            teacher_loss = teacher_loss.sum(dim=-1)
        else:
            # dev or test set
            teacher_loss = 0
        
        if self.cfg.teacher_lambda == 1.0:
            truth_loss = 0.0
        else:
            truth_loss = self.get_nll_loss(model, net_output, sample)

        if isinstance(teacher_loss, torch.Tensor) and isinstance(truth_loss, torch.Tensor):
            assert teacher_loss.shape == truth_loss.shape, (teacher_loss.shape, truth_loss.shape)
        loss = (1.0 - self.cfg.teacher_lambda) * truth_loss + self.cfg.teacher_lambda * teacher_loss

        sample_size = (
            sample["target"].size(0) if self.cfg.sentence_avg else sample["ntokens"]
        )
        
        if self.cfg.teacher_lambda == 1.0 and self.cfg.force_calculate_nll:
            with torch.no_grad():
                truth_loss = self.get_nll_loss(model, net_output, sample)
            
        if reduce:
            loss = loss.sum()
            with torch.no_grad():
                truth_loss = truth_loss.sum() if isinstance(truth_loss, torch.Tensor) else torch.tensor([0.])
                teacher_loss = teacher_loss.sum() if isinstance(teacher_loss, torch.Tensor) else torch.tensor([0.])
        logging_output = {
            "loss": loss.data,
            "nll_loss": truth_loss.data,
            "teacher_loss": teacher_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.cfg.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_nll_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.cfg.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.cfg.ignore_prefix_size:, :].contiguous()
            target = target[:, self.cfg.ignore_prefix_size:].unsqueeze(-1).contiguous()
        truth_loss = -lprobs.gather(dim=-1, index=target)
        pad_mask = target.eq(self.padding_idx)
        truth_loss.masked_fill_(pad_mask, 0.0)
        return truth_loss.view(-1)


    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        teacher_loss_sum = sum(log.get("teacher_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss",  nll_loss_sum / ntokens / math.log(2), round=3
        )
        metrics.log_scalar(
            "truth_loss", nll_loss_sum / sample_size / math.log(2), sample_size, round=3,
        )
        metrics.log_scalar(
            "teacher_loss",
            teacher_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True