import math
import sys
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
import sentencepiece as spm
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import convert_waveform
from fairseq.file_io import PathManager
from typing import Optional

try:
    from simuleval.utils import entrypoint
    from simuleval.agents import TextToTextAgent, SpeechToTextAgent, AgentStates
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.data.segments import Segment, TextSegment, EmptySegment
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

sys.path.append(os.path.dirname(__file__))
from t2t_base_agent import T2TBaseWaitKAgent, T2TAgentStates

BOW_PREFIX = "\u2581"


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

class T2TLocalAgreementAgentStates(T2TAgentStates):


    def reset(self) -> None:
        self.hypos_queue = []
        self.prev_target_length = 0
        return super().reset()

    def update_source(self, segment: Segment, model, gpu, segment_to_units, args, source_segments):
        """
        Update states from input segment
        Additionlly update incremental states
        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished

        if not source_segments == None:
            self.source = segment_to_units(source_segments)

            if args.src_tokens_prefix_tag == "no-eos-no-srclangid":
                self.source_ids = [model.decoder.dictionary.index(i) for i in self.source]
            elif args.src_tokens_prefix_tag == "w-srclangid":
                self.source_ids = [model.decoder.dictionary.index(i) for i in self.source] + [250004]
            elif args.src_tokens_prefix_tag == "w-eos":
                self.source_ids = [model.decoder.dictionary.index(i) for i in self.source] + [model.decoder.dictionary.eos()]
            elif args.src_tokens_prefix_tag == "w-eos-w-srclangid":
                self.source_ids = [model.decoder.dictionary.index(i) for i in self.source] + [model.decoder.dictionary.eos(), 250004]

            self.src_tokens = torch.LongTensor(self.source_ids).cuda().unsqueeze(0)
            self.src_lengths = torch.LongTensor([len(self.source_ids)]).cuda()


class T2TBeamLocalAgreementAgent(T2TBaseWaitKAgent):

    def __init__(self, args):
        super().__init__(args)

        # build generator
        self.source_segments = None
        self.build_generator(args)

    def build_states(self):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        return T2TLocalAgreementAgentStates()

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
        parser.add_argument("--src-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for source text")
        parser.add_argument("--src-splitter-path", type=str, default=None,
                            help="Subword splitter model path for source text")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--tag", type=str, default=None, choices=["off", "si"],
                            help="specifying target's style (Offline or SI)")
        parser.add_argument("--gpu", action="store_true", default=False)
        parser.add_argument("--beam", type=int, default=5, help="beam size")
        parser.add_argument("--initial-wait", type=int, default=0)
        parser.add_argument("--la-n", type=int, default=2,
                            help="agreeing prefixes of n chunks")
        parser.add_argument("--src-tokens-prefix-tag", type=str, help="add eos or langid on tails of src_tokens")
        # parser.add_argument("--source-segment-size", type=int, default=1, help="source tokens segment size")

        # fmt: on
        return parser

    def build_generator(self, args):
        self.loaded_state["cfg"]["generation"]["beam"] = args.beam
        self.loaded_state["cfg"]["generation"]["max_len_b"] = args.max_len
        self.loaded_state["cfg"]["generation"]["prefix_size"] = 0

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

    def policy(self, states):
        if len(self.states.source) <= self.args.initial_wait \
            and not self.states.source_finished:
            return ReadAction()

        tgt_indices = self.to_device(
            torch.LongTensor(
                self.tag_ids
                + [x for x in self.states.target_indices if x is not None]
            ).unsqueeze(0)
        )
        hypos = self.gen_hypos(self.states, tgt_indices)[
            len(self.tag_ids):
        ]
        if not self.states.source_finished:
            # remove EOS + langid
            hypos.pop(-1)
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
            return WriteAction(
                prediction,
                finished=(
                    self.states.target_indices[-1] == self.model.decoder.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )
        return ReadAction()


    def push(
        self, source_segment: Segment, states: Optional[AgentStates] = None
    ) -> None:
        """
        The function to process the incoming information.
        Args:
            source_info (dict): incoming information dictionary
            states (Optional[AgentStates]): an optional states for stateless agent
        """
        if states is None:
            states = self.states

            if source_segment.is_empty:
                pass
            else:
                self.source_segments = source_segment.content

        elif not source_segment.is_empty:
            self.source_segments = " ".join([self.source_segments, source_segment.content])

        if source_segment.is_empty:
            states.update_source(source_segment, self.model, self.gpu, self.segment_to_units, self.args, self.source_segments)
            self.source_segments = None
            
        elif len(self.source_segments.split(" ")) == self.args.source_segment_size:
            states.update_source(source_segment, self.model, self.gpu, self.segment_to_units, self.args, self.source_segments)
            self.source_segments = None

