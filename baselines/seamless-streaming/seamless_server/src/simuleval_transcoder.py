from simuleval.utils.agent import build_system_from_dir
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import soundfile
import io
import asyncio
from simuleval.agents.pipeline import TreeAgentPipeline
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, EmptySegment, SpeechSegment
import threading
import math
import logging
import sys
from pathlib import Path
import time
from g2p_en import G2p
import torch
import traceback
import time
import random
import colorlog

from .speech_and_text_output import SpeechAndTextOutput

MODEL_SAMPLE_RATE = 16_000

logger = logging.getLogger(__name__)
# logger.propagate = False
handler = colorlog.StreamHandler(stream=sys.stdout)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(levelname)s][%(module)s]:%(reset)s %(message)s",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)


class OutputSegments:
    def __init__(self, segments: Union[List[Segment], Segment]):
        if isinstance(segments, Segment):
            segments = [segments]
        self.segments: List[Segment] = [s for s in segments]

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)

    def compute_length(self, g2p):
        lengths = []
        for segment in self.segments:
            if segment.data_type == "text":
                lengths.append(len([x for x in g2p(segment.content) if x != " "]))
            elif segment.data_type == "speech":
                lengths.append(len(segment.content) / MODEL_SAMPLE_RATE)
            elif isinstance(segment, EmptySegment):
                continue
            else:
                logger.warning(
                    f"Unexpected data_type: {segment.data_type} not in 'speech', 'text'"
                )
        return max(lengths)

    @classmethod
    def join_output_buffer(
        cls, buffer: List[List[Segment]], output: SpeechAndTextOutput
    ):
        num_segments = len(buffer[0])
        for i in range(num_segments):
            segment_list = [
                buffer[j][i]
                for j in range(len(buffer))
                if buffer[j][i].data_type is not None
            ]
            if len(segment_list) == 0:
                continue
            if len(set(segment.data_type for segment in segment_list)) != 1:
                logger.warning(
                    f"Data type mismatch at {i}: {set(segment.data_type for segment in segment_list)}"
                )
                continue
            data_type = segment_list[0].data_type
            if data_type == "text":
                if output.text is not None:
                    logger.warning("Multiple text outputs, overwriting!")
                output.text = " ".join([segment.content for segment in segment_list])
            elif data_type == "speech":
                if output.speech_samples is not None:
                    logger.warning("Multiple speech outputs, overwriting!")
                speech_out = []
                for segment in segment_list:
                    speech_out += segment.content
                output.speech_samples = speech_out
                output.speech_sample_rate = segment.sample_rate
            elif isinstance(segment_list[0], EmptySegment):
                continue
            else:
                logger.warning(
                    f"Invalid output buffer data type: {data_type}, expected 'speech' or 'text"
                )

        return output

    def __repr__(self) -> str:
        repr_str = str(self.segments)
        return f"{self.__class__.__name__}(\n\t{repr_str}\n)"


class SimulevalTranscoder:
    def __init__(self, agent, sample_rate, debug, buffer_limit):
        self.agent = agent.agent
        self.has_expressive = agent.has_expressive
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.states = self.agent.build_states()
        if debug:
            self.get_states_root().debug = True
        self.incoming_sample_rate = sample_rate
        self.close = False
        self.g2p = G2p()

        # buffer all outgoing translations within this amount of time
        self.output_buffer_idle_ms = 5000
        self.output_buffer_size_limit = (
            buffer_limit  # phonemes for text, seconds for speech
        )
        self.output_buffer_cur_size = 0
        self.output_buffer: List[List[Segment]] = []
        self.speech_output_sample_rate = None

        self.last_output_ts = time.time() * 1000
        self.timeout_ms = (
            30000  # close the transcoder thread after this amount of silence
        )
        self.first_input_ts = None
        self.first_output_ts = None
        self.debug = debug
        self.debug_ts = f"{time.time()}_{random.randint(1000, 9999)}"
        if self.debug:
            debug_folder = Path(__file__).resolve().parent.parent / "debug"
            self.test_incoming_wav = soundfile.SoundFile(
                debug_folder / f"{self.debug_ts}_test_incoming.wav",
                mode="w+",
                format="WAV",
                subtype="PCM_16",
                samplerate=self.incoming_sample_rate,
                channels=1,
            )
            self.get_states_root().test_input_segments_wav = soundfile.SoundFile(
                debug_folder / f"{self.debug_ts}_test_input_segments.wav",
                mode="w+",
                format="WAV",
                samplerate=MODEL_SAMPLE_RATE,
                channels=1,
            )

    def get_states_root(self) -> AgentStates:
        if isinstance(self.agent, TreeAgentPipeline):
            # self.states is a dict
            return self.states[self.agent.source_module]
        else:
            # self.states is a list
            return self.states[0]

    def reset_states(self):
        if isinstance(self.agent, TreeAgentPipeline):
            states_iter = self.states.values()
        else:
            states_iter = self.states
        for state in states_iter:
            state.reset()

    def debug_log(self, *args):
        if self.debug:
            logger.info(*args)

    @classmethod
    def build_agent(cls, model_path, config_name):
        logger.info(f"Building simuleval agent: {model_path}, {config_name}")
        agent = build_system_from_dir(
            Path(__file__).resolve().parent.parent / f"models/{model_path}",
            config_name=config_name,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.to(device, fp16=True)
        logger.info(
            f"Successfully built simuleval agent {model_path} on device {device}"
        )

        return agent

    def process_incoming_bytes(self, incoming_bytes, dynamic_config):
        # TODO: We probably want to do some validation on dynamic_config to ensure it has what we needs
        segment, sr = self._preprocess_wav(incoming_bytes)
        segment = SpeechSegment(
            content=segment,
            sample_rate=sr,
            tgt_lang=dynamic_config.get("targetLanguage"),
            config=dynamic_config,
        )
        if dynamic_config.get("expressive") is True and self.has_expressive is False:
            logger.warning(
                "Passing 'expressive' but the agent does not support expressive output!"
            )
        # # segment is array([0, 0, 0, ..., 0, 0, 0], dtype=int16)
        self.input_queue.put_nowait(segment)

    def get_input_segment(self):
        if self.input_queue.empty():
            return None
        chunk = self.input_queue.get_nowait()
        self.input_queue.task_done()
        return chunk

    def convert_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        normalize_volume: bool = False,
        to_mono: bool = False,
        to_sample_rate: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
        """convert a waveform:
        - to a target sample rate
        - from multi-channel to mono channel
        - volume normalization

        Args:
            waveform (numpy.ndarray or torch.Tensor): 2D original waveform
                (channels x length)
            sample_rate (int): original sample rate
            normalize_volume (bool): perform volume normalization
            to_mono (bool): convert to mono channel if having multiple channels
            to_sample_rate (Optional[int]): target sample rate
        Returns:
            waveform (numpy.ndarray): converted 2D waveform (channels x length)
            sample_rate (float): target sample rate
        """
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError("Please install torchaudio: pip install torchaudio")

        effects = []
        if normalize_volume:
            effects.append(["gain", "-n"])
        if to_sample_rate is not None and to_sample_rate != sample_rate:
            effects.append(["rate", f"{to_sample_rate}"])
        if to_mono and waveform.shape[0] > 1:
            effects.append(["channels", "1"])
        if len(effects) > 0:
            is_np_input = isinstance(waveform, np.ndarray)
            _waveform = torch.from_numpy(waveform) if is_np_input else waveform
            converted, converted_sample_rate = ta_sox.apply_effects_tensor(
                _waveform, sample_rate, effects
            )
            if is_np_input:
                converted = converted.numpy()
            return converted, converted_sample_rate
        return waveform, sample_rate

    def _preprocess_wav(self, data: Any) -> Tuple[np.ndarray, int]:
        segment, sample_rate = soundfile.read(
            io.BytesIO(data),
            dtype="float32",
            always_2d=True,
            frames=-1,
            start=0,
            format="RAW",
            subtype="PCM_16",
            samplerate=self.incoming_sample_rate,
            channels=1,
        )
        if self.debug:
            self.test_incoming_wav.seek(0, soundfile.SEEK_END)
            self.test_incoming_wav.write(segment)

        segment = segment.T
        segment, new_sample_rate = self.convert_waveform(
            segment,
            sample_rate,
            normalize_volume=False,
            to_mono=True,
            to_sample_rate=MODEL_SAMPLE_RATE,
        )

        assert MODEL_SAMPLE_RATE == new_sample_rate
        segment = segment.squeeze(axis=0)
        return segment, new_sample_rate

    def process_pipeline_impl(self, input_segment):
        try:
            with torch.no_grad():
                output_segment = OutputSegments(
                    self.agent.pushpop(input_segment, self.states)
                )
            if (
                self.get_states_root().first_input_ts is not None
                and self.first_input_ts is None
            ):
                # TODO: this is hacky
                self.first_input_ts = self.get_states_root().first_input_ts

            if not output_segment.is_empty:
                self.output_queue.put_nowait(output_segment)

            if output_segment.finished:
                self.debug_log("OUTPUT SEGMENT IS FINISHED. Resetting states.")

                self.reset_states()

                if self.debug:
                    # when we rebuild states, this value is reset to whatever
                    # is in the system dir config, which defaults debug=False.
                    self.get_states_root().debug = True
        except Exception as e:
            logger.error(f"Got exception while processing pipeline: {e}")
            traceback.print_exc()
        return input_segment

    def process_pipeline_loop(self):
        if self.close:
            return  # closes the thread

        self.debug_log("processing_pipeline")
        while not self.close:
            input_segment = self.get_input_segment()
            if input_segment is None:
                if self.get_states_root().is_fresh_state:  # TODO: this is hacky
                    time.sleep(0.3)
                else:
                    time.sleep(0.03)
                continue
            self.process_pipeline_impl(input_segment)
        self.debug_log("finished processing_pipeline")

    def process_pipeline_once(self):
        if self.close:
            return

        self.debug_log("processing pipeline once")
        input_segment = self.get_input_segment()
        if input_segment is None:
            return
        self.process_pipeline_impl(input_segment)
        self.debug_log("finished processing_pipeline_once")

    def get_output_segment(self):
        if self.output_queue.empty():
            return None

        output_chunk = self.output_queue.get_nowait()
        self.output_queue.task_done()
        return output_chunk

    def start(self):
        self.debug_log("starting transcoder in a thread")
        threading.Thread(target=self.process_pipeline_loop).start()

    def first_translation_time(self):
        return round((self.first_output_ts - self.first_input_ts) / 1000, 2)

    def get_buffered_output(self) -> SpeechAndTextOutput:
        now = time.time() * 1000
        self.debug_log(f"get_buffered_output queue size: {self.output_queue.qsize()}")
        while not self.output_queue.empty():
            tmp_out = self.get_output_segment()
            if tmp_out and tmp_out.compute_length(self.g2p) > 0:
                if len(self.output_buffer) == 0:
                    self.last_output_ts = now
                self._populate_output_buffer(tmp_out)
                self._increment_output_buffer_size(tmp_out)

                if tmp_out.finished:
                    self.debug_log("tmp_out.finished")
                    res = self._gather_output_buffer_data(final=True)
                    self.debug_log(f"gathered output data: {res}")
                    self.output_buffer = []
                    self.increment_output_buffer_size = 0
                    self.last_output_ts = now
                    self.first_output_ts = now
                    return res
            else:
                self.debug_log("tmp_out.compute_length is not > 0")

        if len(self.output_buffer) > 0 and (
            now - self.last_output_ts >= self.output_buffer_idle_ms
            or self.output_buffer_cur_size >= self.output_buffer_size_limit
        ):
            self.debug_log(
                "[get_buffered_output] output_buffer is not empty. getting res to return."
            )
            self.last_output_ts = now
            res = self._gather_output_buffer_data(final=False)
            self.debug_log(f"gathered output data: {res}")
            self.output_buffer = []
            self.output_buffer_phoneme_count = 0
            self.first_output_ts = now
            return res
        else:
            self.debug_log("[get_buffered_output] output_buffer is empty...")
            return None

    def _gather_output_buffer_data(self, final):
        output = SpeechAndTextOutput()
        output.final = final
        output = OutputSegments.join_output_buffer(self.output_buffer, output)
        return output

    def _increment_output_buffer_size(self, segment: OutputSegments):
        self.output_buffer_cur_size += segment.compute_length(self.g2p)

    def _populate_output_buffer(self, segment: OutputSegments):
        self.output_buffer.append(segment.segments)

    def _compute_phoneme_count(self, string: str) -> int:
        return len([x for x in self.g2p(string) if x != " "])
