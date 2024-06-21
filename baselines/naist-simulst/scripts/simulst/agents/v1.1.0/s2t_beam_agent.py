import math
import sys
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
import yaml
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import convert_waveform
from fairseq.file_io import PathManager
from typing import Optional
import logging

try:
    from simuleval.utils import entrypoint
    from simuleval.agents import SpeechToTextAgent, AgentStates
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.data.segments import Segment, TextSegment, EmptySegment
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

sys.path.append(os.path.dirname(__file__))
from s2t_base_agent import S2TBaseWaitKAgent, S2TAgentStates

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
LANG_TOKENS = ["<lang:de>", "<lang:ja>", "<lang:zh>"]


def longest_common_prefix(strs):

    n_str = len(strs)

    if n_str == 1:
        return strs[0]

    # find LCP
    strs.sort()
    end = min(len(strs[0]), len(strs[-1]))
    i = 0
    while i < end and strs[0][i] == strs[-1][i]:
        i += 1

    return strs[0][:i]


class S2TLocalAgreementAgentStates(S2TAgentStates):
    def reset(self) -> None:
        self.hypos_queue = []
        self.prev_target_length = 0
        self.ngram_counts = {}
        return super().reset()

    def update_source(self, segment: Segment, model, gpu, feature_extractor):
        """
        Update states from input segment
        Additionlly update incremental states
        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if not segment.is_empty:
            if len(self.source) > 0:
                self.source = torch.cat((self.source, feature_extractor(segment.content)))
            else:
                self.source = feature_extractor(segment.content)

            self.src_tokens = torch.FloatTensor(self.source).cuda().unsqueeze(0)
            self.src_lengths = torch.LongTensor([len(self.source)]).cuda()
            

class S2TBeamLocalAgreementAgent(S2TBaseWaitKAgent):

    def __init__(self, args):
        super().__init__(args)

        # build generator
        self.build_generator(args)

    def build_states(self):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        return S2TLocalAgreementAgentStates()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--use-audio-input", action="store_true")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--tag", type=str, default=None, choices=["off", "si"],
                            help="specifying target's style (Offline or SI)")
        parser.add_argument("--gpu", action="store_true", default=False)
        parser.add_argument("--beam", type=int, default=5, help="beam size")
        parser.add_argument("--initial-wait", type=int, default=0)
        parser.add_argument("--la-n", type=int, default=2,
                            help="agreeing prefixes of n chunks")

        # fmt: on
        return parser

    def build_generator(self, args):
        self.loaded_state["cfg"]["generation"]["beam"] = args.beam
        self.loaded_state["cfg"]["generation"]["max_len_b"] = args.max_len
        self.loaded_state["cfg"]["generation"]["prefix_size"] = 1

        # build generator
        self.generator = self.task.build_generator(
            [self.model], self.loaded_state["cfg"]["generation"]
        )

    def gen_hypos(self, states, prefix_tokens):
        sample = {
            "net_input": {
                "src_tokens": states.src_tokens,
                "src_lengths": states.src_lengths,
            },
        }
        hypos = self.task.inference_step(
            self.generator,
            self.model,
            sample,
            prefix_tokens = prefix_tokens,
        )
        return hypos[0][0]["tokens"].tolist()

    def repetition_check(self, prediction, n=3, count_max=3):

        words = prediction.split()

        for i in range(len(words)-n+1):
            ngram = tuple(words[i:i+n])
            if self.states.ngram_counts.get(ngram):
                self.states.ngram_counts[ngram] += 1
            else:
                self.states.ngram_counts[ngram] = 1

        if len(self.states.ngram_counts) == 0:
            return False
        if max(self.states.ngram_counts.values()) >= count_max:
            return True
        return False

    def remove_strings(self, text, exclude_list):
        for exclude_str in exclude_list:
            text = text.replace(exclude_str, "")
        return text

    def policy(self, states):
        if 1000 * len(self.states.source) / 16000 <= self.args.initial_wait \
            and not self.states.source_finished:
            return ReadAction()

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.lang_id]
                + self.tag_ids
                + [x for x in self.states.target_indices if x is not None]
            ).unsqueeze(0)
        )

        hypos = self.gen_hypos(self.states, tgt_indices)[
            len([self.lang_id] + self.tag_ids):
        ]
        if not self.states.source_finished:
            # remove EOS
            hypos.pop(-1)
        torch.cuda.empty_cache()

        self.states.hypos_queue.append(hypos)

        # local agreement
        if len(self.states.hypos_queue) >= self.args.la_n \
            and not self.states.source_finished:
            self.states.target_indices = longest_common_prefix(
                self.states.hypos_queue.copy()
            )
            self.states.hypos_queue.pop(0)

        elif self.states.source_finished:
            self.states.target_indices = self.states.hypos_queue[-1]

        else:
            return ReadAction()

        if len(self.states.target_indices) > 1:
            # remove tokens already written
            prediction = self.model.decoder.dictionary.string(
                self.states.target_indices[self.states.prev_target_length:]
            )

            self.states.prev_target_length = len(self.states.target_indices)
            is_end = self.repetition_check(prediction, n=3, count_max=3)
            return WriteAction(
                prediction,
                finished=(
                    self.states.target_indices[-1] == self.model.decoder.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )
        return ReadAction()


class S2TWithEDAttAgent(S2TBeamLocalAgreementAgent):

    def __init__(self, args):
        super().__init__(args)

        # build generator
        self.build_generator(args)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--use-audio-input", action="store_true")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--tag", type=str, default=None, choices=["off", "si"],
                            help="specifying target's style (Offline or SI)")
        parser.add_argument("--gpu", action="store_true", default=False)
        parser.add_argument("--beam", type=int, default=5, help="beam size")
        parser.add_argument("--initial-wait", type=int, default=0)
        parser.add_argument("--attn-threshold", type=float, default=0.1,
                            help="Threshold on the attention scores that triggers READ action."
                                 "If the last frame attention score >= attn_threshold, READ action is performed.")
        parser.add_argument("--frame-num", default=1, type=int,
                            help="Number of frames to consider for the attention scores starting from the end.")
        parser.add_argument("--exclude-last-attn", action="store_true", default=False,
                            help="Exclude last attention score from the average.")

    def gen_hypos_with_attn(self, states, prefix_tokens):
        sample = {
            "net_input": {
                "src_tokens": states.src_tokens,
                "src_lengths": states.src_lengths,
            },
        }
        hypos = self.task.inference_step(
            self.generator,
            self.model,
            sample,
            prefix_tokens = prefix_tokens,
        )
        return hypos[0][0]["tokens"].int().cpu(), hypos[0][0]["attention"].float().cpu()

    def policy(self, states):
        if 1000 * len(self.states.source) / 16000 <= self.args.initial_wait \
            and not self.states.source_finished:
            return ReadAction()

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.lang_id]
                + self.tag_ids
                + [x for x in self.states.target_indices if x is not None]
            ).unsqueeze(0)
        )

        # get hypothesis and attention scores
        hypo_tokens, hypo_attn = self.gen_hypos_with_attn(self.states, tgt_indices)
        if not self.states.source_finished:
            # remove EOS
            hypo_tokens = hypo_tokens[:-1]
            hypo_attn = hypo_attn[:, :-1]
        torch.cuda.empty_cache()
        assert hypo_tokens.size(0) == hypo_attn.size(1)

        # EDAtt
        # select new partial hypothesis (without the prefix tokens and the already emitted tokens)
        prefix_len = tgt_indices.size(1)
        new_hypo = hypo_tokens[prefix_len:]
        hypo_attn = hypo_attn[:, prefix_len:]

        if not self.states.source_finished:
            if self.args.exclude_last_attn:
                # normalize considering all but last attention score (original paper)
                normalized_attn = F.normalize(hypo_attn[:-1, :].transpose(0, 1), dim=1)
            else:
                # normalize considering all attention scores
                normalized_attn = F.normalize(hypo_attn.transpose(0, 1), dim=1)
            if normalized_attn.size(0) > 0 and normalized_attn.size(1) > 0:
                # find which tokens rely on the last n frames using threshold
                curr_attn = normalized_attn[:, -self.args.frame_num:]
                last_frames_attn = torch.sum(curr_attn, dim=1)
                # for each element of the tensor, we check if the sum exceeds or is equal to the attn_threshold,
                # we find the list of indexes for which this is True (corresponding to 1 value) by applying
                # the nonzero() function, and we select the first token for which the threshold has been
                # exceeded, corresponding to the index from which the emission is stopped
                invalid_token_idxs = (last_frames_attn >= self.args.attn_threshold).nonzero(as_tuple=True)[0]
                if len(invalid_token_idxs) > 0:
                    # if there are tokens that exceed the threshold, we select the first one
                    # and we remove the corresponding tokens from the new hypothesis
                    invalid_token_idx = invalid_token_idxs[0]
                    new_hypo = new_hypo[:invalid_token_idx]
                if len(new_hypo) > 0:
                    # if there are tokens in the new hypothesis, we add them to the target indices
                    # and we update the target indices
                    self.states.target_indices += new_hypo.tolist()
        elif self.states.source_finished:
            self.states.target_indices += new_hypo.tolist()

        if len(self.states.target_indices) > 1:
            # remove tokens already written
            prediction = self.model.decoder.dictionary.string(
                self.states.target_indices[self.states.prev_target_length:]
            )
            self.states.prev_target_length = len(self.states.target_indices)
            return WriteAction(
                prediction,
                finished=(
                    self.states.target_indices[-1] == self.model.decoder.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )
        return ReadAction()

class S2TWithAlignAttAgent(S2TBeamLocalAgreementAgent):

    def __init__(self, args):
        super().__init__(args)

        # build generator
        self.build_generator(args)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--use-audio-input", action="store_true")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--tag", type=str, default=None, choices=["off", "si"],
                            help="specifying target's style (Offline or SI)")
        parser.add_argument("--gpu", action="store_true", default=False)
        parser.add_argument("--beam", type=int, default=5, help="beam size")
        parser.add_argument("--initial-wait", type=int, default=0)
        parser.add_argument("--frame-num", default=1, type=int,
                            help="Number of the most recent frames to which"
                            " a token predominantly attends before the emission is halted")
        parser.add_argument("--exclude-first-attn", action="store_true", default=False,
                            help="Exclude first attention score from the alignment.")

    def gen_hypos_with_attn(self, states, prefix_tokens):
        sample = {
            "net_input": {
                "src_tokens": states.src_tokens,
                "src_lengths": states.src_lengths,
            },
        }
        hypos = self.task.inference_step(
            self.generator,
            self.model,
            sample,
            prefix_tokens = prefix_tokens,
        )
        return hypos[0][0]["tokens"].int().cpu(), hypos[0][0]["attention"].float().cpu()

    def repetition_check(self, prediction, n=3, count_max=3):

        words = prediction.split()

        for i in range(len(words)-n+1):
            ngram = tuple(words[i:i+n])
            if self.states.ngram_counts.get(ngram):
                self.states.ngram_counts[ngram] += 1
            else:
                self.states.ngram_counts[ngram] = 1

        if len(self.states.ngram_counts) == 0:
            return False
        if max(self.states.ngram_counts.values()) >= count_max:
            return True
        return False

    def remove_strings(self, text, exclude_list):
        for exclude_str in exclude_list:
            text = text.replace(exclude_str, "")
        return text

    def policy(self, states):
        if 1000 * len(self.states.source) / 16000 <= self.args.initial_wait \
            and not self.states.source_finished:
            return ReadAction()

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.lang_id]
                + self.tag_ids
                + [x for x in self.states.target_indices if x is not None]
            ).unsqueeze(0)
        )

        # get hypothesis and attention scores
        hypo_tokens, hypo_attn = self.gen_hypos_with_attn(self.states, tgt_indices)
        if not self.states.source_finished:
            # remove EOS
            hypo_tokens = hypo_tokens[:-1]
            hypo_attn = hypo_attn[:, :-1]
        torch.cuda.empty_cache()
        assert hypo_tokens.size(0) == hypo_attn.size(1)

        # Alignatt
        # select new partial hypothesis (without the prefix tokens and the already emitted tokens)
        prefix_len = tgt_indices.size(1)
        new_hypo = hypo_tokens[prefix_len:]
        hypo_attn = hypo_attn[:, prefix_len:]

        if not self.states.source_finished:
            if self.args.exclude_first_attn:
                # normalize considering all but first attention score
                # Empirically, the first attention tends to be focused, so exclude the first attention
                normalized_attn = F.normalize(hypo_attn[1:, :].transpose(0, 1), dim=1)
            else:
                # normalize considering all attention scores
                normalized_attn = F.normalize(hypo_attn.transpose(0, 1), dim=1)

            if normalized_attn.size(0) > 0 and normalized_attn.size(1) > 0:
                # get tgt-src alignment vector
                alignment = torch.argmax(normalized_attn, dim=1)
                # checking wheather each tgt token is aligned to the last n frames or not
                # stop the emission as soon as we find a token that mostly attends to the last n frames
                invalid_token_idxs = (alignment >= normalized_attn.size(1) - self.args.frame_num).nonzero(as_tuple=True)[0]

                if len(invalid_token_idxs) > 0:
                    invalid_token_idx = invalid_token_idxs[0]
                    new_hypo = new_hypo[:invalid_token_idx]
                if len(new_hypo) > 0:
                    # if there are tokens in the new hypothesis, we add them to the target indices
                    # and we update the target indices
                    self.states.target_indices += new_hypo.tolist()
        elif self.states.source_finished:
            self.states.target_indices += new_hypo.tolist()

        def string_to_indices(string):
            return " ".join(
                map(str,
                    self.model.decoder.dictionary.encode_line(
                        string, append_eos=False, add_if_not_exist=False
                    ).long().tolist()
                )
            )

        if len(self.states.target_indices) > 1:
            # remove tokens already written
            prediction = self.model.decoder.dictionary.string(
                self.states.target_indices[self.states.prev_target_length:]
            )

            self.states.prev_target_length = len(self.states.target_indices)

            is_end = self.repetition_check(prediction, n=3, count_max=3)
            if is_end:
                return WriteAction(
                   prediction,
                   finished=True,
                )

            return WriteAction(
                prediction,
                finished=(
                    self.states.target_indices[-1] == self.model.decoder.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )
        return ReadAction()
