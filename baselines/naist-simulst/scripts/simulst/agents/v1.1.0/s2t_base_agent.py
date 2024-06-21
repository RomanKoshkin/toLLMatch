import math
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import convert_waveform
from fairseq.file_io import PathManager
from typing import Optional

try:
    from simuleval.utils import entrypoint
    from simuleval.agents import SpeechToTextAgent, AgentStates
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.data.segments import Segment, TextSegment, EmptySegment
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
LANG_TOKENS = ["<lang:de>", "<lang:ja>", "<lang:zh>"]


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.use_audio_input = args.use_audio_input

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        if self.use_audio_input:
            output = torch.Tensor(new_samples)
        else:
            samples = self.previous_residual_samples + new_samples
            if len(samples) < self.num_samples_per_window:
                self.previous_residual_samples = samples
                return
 
            # num_frames is the number of frames from the new segment
            num_frames = math.floor(
                (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
                / self.num_samples_per_shift
            )
 
            # the number of frames used for feature extraction
            # including some part of the previous segment
            effective_num_samples = int(
                num_frames * self.len_ms_to_samples(self.shift_size)
                + self.len_ms_to_samples(self.window_size - self.shift_size)
            )
 
            input_samples = samples[:effective_num_samples]
            self.previous_residual_samples = samples[
                num_frames * self.num_samples_per_shift:
            ]

            torch.manual_seed(1)
            output = kaldi.fbank(
                torch.FloatTensor(input_samples).unsqueeze(0),
                num_mel_bins=self.feature_dim,
                frame_length=self.window_size,
                frame_shift=self.shift_size,
            ).numpy()

        output = self.transform(output)

        return output

    def transform(self, input):
        if self.global_cmvn is None:
            mean = input.mean(axis=0)
            square_sums = (input**2).sum(axis=0)
            var = square_sums / input.shape[0] - mean**2
            std = np.sqrt(np.maximum(var, 1e-10))
        else:
            mean = self.global_cmvn["mean"]
            std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


class S2TAgentStates(AgentStates):
    def reset(self) -> None:
        self.incremental_states = {"steps": {"src": 0, "tgt": 1}}
        self.encoder_states = None
        self.target_indices = []
        self.sorce = torch.Tensor()
        return super().reset()

    # [TODO] refactoring
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
            src_indices = torch.FloatTensor(self.source).cuda().unsqueeze(0)
            src_lengths = torch.LongTensor([len(self.source)]).cuda()
            
            torch.cuda.empty_cache()
            self.encoder_states = model.encoder(src_indices, src_lengths)

            self.incremental_states["steps"] = {
                "src": self.encoder_states["encoder_out"][0].size(0),
                "tgt": 1 + len(self.target_indices),
            }

            self.incremental_states["online"] = {"only": torch.tensor(not segment.finished)}


class S2TBaseWaitKAgent(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)

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

        self.feature_extractor = OnlineFeatureExtractor(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

    def build_states(self):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        return S2TAgentStates()

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

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
        parser.add_argument("--waitk", type=int, default=3)

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
            self.lang_id = self.model.decoder.dictionary.index(f"<lang:{args.lang}>")

        self.tag_ids = []
        if args.tag is not None:
            self.tag_ids = [
              self.model.decoder.dictionary.index(i) for i in [chr(int("2581", 16))+"<", f"{args.tag}", ">"]
            ]

    def policy(self, states):
        source_segment_length = int(1000 * len(self.states.source) / (16000 * self.source_segment_size))
        lagging = source_segment_length - len(self.states.target)

        if lagging >= self.args.waitk or self.states.source_finished:

            tgt_indices = self.to_device(
                torch.LongTensor(
                    [self.model.decoder.dictionary.eos()]
                    + [self.lang_id]
                    + self.tag_ids
                    + [x for x in self.states.target_indices if x is not None]
                ).unsqueeze(0)
            )

            x, outputs = self.model.decoder.forward(
                prev_output_tokens=tgt_indices,
                encoder_out=self.states.encoder_states,
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
        states.update_source(source_segment, self.model, self.gpu, self.feature_extractor)
