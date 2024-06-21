import math
import os
import sys
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.audio.audio_utils import convert_waveform
from fairseq.file_io import PathManager
from fairseq_cli.generate import get_symbols_to_strip_from_output

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

sys.path.append(os.path.dirname(__file__))
from simulst_base_agent import SimulSTBaseAgent, TensorListEntry

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
LANG_TOKENS = ["<lang:de>", "<lang:ja>", "<lang:zh>"]


class SimulSTBeamSearchAgent(SimulSTBaseAgent):

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
        parser.add_argument("--chunk-size", type=int, default=250,
                            help="fixed chunk size of speech (in ms)")
        parser.add_argument("--lang", type=str, default=None, choices=["de", "ja", "zh"],
                            help="specifying target language for mBART")
        parser.add_argument("--beam", type=int, default=5, help="beam size")

        # fmt: on
        return parser

    def build_generator(self, args):
        self.loaded_state["cfg"]["generation"]["beam"] = args.beam
        self.loaded_state["cfg"]["generation"]["max_len_b"] = args.max_len
        self.loaded_state["cfg"]["generation"]["prefix_size"] = 1  # TMP?

        # build generator
        self.generator = self.task.build_generator(
            [self.model], self.loaded_state["cfg"]["generation"]
        )

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.prediction_queue = []

    def update_model_encoder(self, states):
            return

    def gen_hypos(self, states, prefix_tokens):
        src_tokens = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
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
        if not states.finish_read():
            return READ_ACTION
        else:
            if states.prediction_queue != []:
                return WRITE_ACTION

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
            states.prediction_queue = gen_hypos(states, tgt_indices)

            torch.cuda.empty_cache()

            return WRITE_ACTION

    def predict(self, states):
        if states.prediction_queue != []:
            return states.prediction_queue.pop(0)
