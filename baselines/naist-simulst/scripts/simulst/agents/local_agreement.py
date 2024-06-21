import math
import os
import sys
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import convert_waveform
from fairseq.file_io import PathManager

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

sys.path.append(os.path.dirname(__file__))
from simulst_base_agent import TensorListEntry
from simulst_beam_agent import SimulSTBeamSearchAgent

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


class SimulSTLocalAgreementAgent(SimulSTBeamSearchAgent):

    def __init__(self, args):
        super().__init__(args)

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
        parser.add_argument("--chunk-size", type=int, default=500,
                            help="fixed chunk size of speech (in ms)")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--beam", type=int, default=5, help="beam size")
        parser.add_argument("--initial-wait", type=int, default=0)
        parser.add_argument("--la-n", type=int, default=2,
                            help="agreeing prefixes of n chunks")

        # fmt: on
        return parser

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.writing = False
        states.prediction_queue = []
        states.hypos_queue = []

    def policy(self, states):
        if states.num_milliseconds() <= self.args.initial_wait \
                and not states.finish_read():
            return READ_ACTION
        else:
            # Keep writing
            if states.prediction_queue != []:
                states.writing = True
                return WRITE_ACTION
            # Turn to read
            elif not states.finish_read() and states.writing:
                states.writing = False
                return READ_ACTION

            # Decode and write

            # specify target language id for multilingual decoder
            lang_id = None
            if self.args.lang is not None:
                if self.args.lang == "de":
                    lang_id = self.model.decoder.dictionary.index("<lang:de>")
                elif self.args.lang == "ja":
                    lang_id = self.model.decoder.dictionary.index("<lang:ja>")
                elif self.args.lang == "zh":
                    lang_id = self.model.decoder.dictionary.index("<lang:zh>")

            # set prefix tokens
            tgt_indices = [x for x in states.units.target.value if x is not None]
            if lang_id:
                tgt_indices = [lang_id] + tgt_indices
            tgt_indices = self.to_device(
                torch.LongTensor(tgt_indices).unsqueeze(0)
            )

            # translate whole speech with beam search
            hypos = self.gen_hypos(states, tgt_indices)
            if not states.finish_read():
                # remove EOS
                hypos.pop(-1)

            states.hypos_queue.append(hypos)

            # local agreement
            if len(states.hypos_queue) >= self.args.la_n \
                and not states.finish_read():
                states.prediction_queue = longest_common_prefix(states.hypos_queue.copy())
                states.hypos_queue.pop(0)

            elif states.finish_read():
                states.prediction_queue = states.hypos_queue[-1]
                    
            # remove tokens already written
            states.prediction_queue = states.prediction_queue[tgt_indices.size(-1):]

            torch.cuda.empty_cache()

            states.writing = True
            return WRITE_ACTION
