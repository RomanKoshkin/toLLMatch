import math
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

BOW_PREFIX = "\u2581"

class T2TAgentStates(AgentStates):
    def reset(self) -> None:
        self.incremental_states = {"steps": {"src": 0, "tgt": 1}}
        self.encoder_states = None
        self.target_indices = []
        self.sorce = torch.Tensor()
        self.source_segments = ""
        return super().reset()

    # # [TODO] refactoring
    # def update_source(self, segment: Segment, model, gpu, segment_to_units, args):
    #     """
    #     Update states from input segment
    #     Additionlly update incremental states
    #     Args:
    #         segment (~simuleval.agents.segments.Segment): input segment
    #     """
    #     # segment -> subword
    #     # segment + segment -> subword -> self.source
    #     self.source_finished = segment.finished
    #     if not segment.is_empty:
    #         if len(self.source) > 0:
    #             self.source = self.source + [segment.content]
    #         else:
    #             self.source = [segment.content]
    #
    #         self.source = " ".join(self.source)
    #         self.source = segment_to_units(self.source)
    #

    # def update_source(self, segment: Segment, model, gpu, segment_to_units, args, source_segments):
    def update_source(self, segment: Segment, model, gpu, args, segment_to_units):
        """
        Update states from input segment
        Additionlly update incremental states
        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
##
        if not segment.is_empty:
            if len(self.source) > 0:
                # self.source = self.source + segment_to_units(segment.content)
                self.source += [segment.content]
            else:
                self.source = [segment.content]

            self.source = segment_to_units(" ".join(self.source))

            # [FIXME] only for english
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
            
            self.encoder_states = model.encoder(self.src_tokens, self.src_lengths)

            self.incremental_states["steps"] = {
                "src": self.encoder_states["encoder_out"][0].size(0),
                "tgt": 1 + len(self.target_indices),
            }

            self.incremental_states["online"] = {"only": torch.tensor(not segment.finished)}
            torch.cuda.empty_cache()


class T2TBaseWaitKAgent(TextToTextAgent):

    def __init__(self, args):
        super().__init__(args)

        self.source_segments = None
        self.source_segment_size = args.source_segment_size

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        # load model and vocabulary
        self.load_model_vocab(args)

        args.global_cmvn = None
        if args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        if args.global_stats:
            with PathManager.open(args.global_stats, "r") as f:
                global_cmvn = json.loads(f.read())
                self.global_cmvn = {"mean": global_cmvn["mean"], "std": global_cmvn["stddev"]}

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

        self.build_word_splitter(args)



    def build_states(self):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        return T2TAgentStates()

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def build_word_splitter(self, args):
        self.spm = {}
        for lang in ['src', 'tgt']:
            if getattr(args, f'{lang}_splitter_type', None):
                path = getattr(args, f'{lang}_splitter_path', None)
                if path:
                    self.spm[lang] = spm.SentencePieceProcessor()
                    self.spm[lang].Load(path)

    def segment_to_units(self, segment):
        # Split a full word (segment) into subwords (units)
        return self.spm['src'].EncodeAsPieces(segment)

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
        parser.add_argument("--waitk", type=int, default=3)
        parser.add_argument("--src-tokens-prefix-tag", type=str, help="add eos or langid on tails of src_tokens")

        # fmt: on
        return parser

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        self.loaded_state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        # override task.data
        task_args = self.loaded_state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config is not None:
            task_args.data_config_yaml = args.config

        self.task = tasks.setup_task(task_args)

        # build model for ensemble
        self.model = self.task.build_model(self.loaded_state["cfg"]["model"])
        self.model.load_state_dict(self.loaded_state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = self.task.target_dictionary

        if args.lang is not None:
            # [INFO] mbart trained t2t
            self.lang_id = self.model.decoder.dictionary.index(f"[{args.lang}]")


        self.tag_ids = []
        if args.tag is not None:
            self.tag_ids = [
              self.model.decoder.dictionary.index(i) for i in [chr(int("2581", 16))+"<", f"{args.tag}", ">"]
            ]

    def policy(self, states):
        lagging = len(self.states.source) - len(self.states.target)

        if lagging >= self.args.waitk or self.states.source_finished:

            tgt_indices = self.to_device(
                torch.LongTensor(
                    [self.lang_id]
                    + self.tag_ids
                    + [x for x in self.states.target_indices if x is not None]
                ).unsqueeze(0)
            )
                    # [self.model.decoder.dictionary.eos()]

            x, outputs = self.model.decoder.forward(
                prev_output_tokens=tgt_indices,
                encoder_out=self.states.encoder_states,
                #incremental_state=states.incremental_states,
            )
    
            states.decoder_out = x
    
            torch.cuda.empty_cache()

            log_probs = self.model.get_normalized_probs(
                [states.decoder_out[:, -1:]], log_probs=True
            )
            index = log_probs.argmax(dim=-1)[0, 0].item()

            self.states.target_indices.append(index)

            return WriteAction(
                self.model.decoder.dictionary.string([index]),
                finished=(
                    index == self.model.decoder.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )

        else:
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
        states.update_source(source_segment, self.model, self.gpu, self.args, self.segment_to_units)
