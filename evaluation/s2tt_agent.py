"""
3 ASR classes
    Asr - uses HuggingFace Whisper model (slow)
    AsrJAX - uses Whiseper-JAX either locally or through an API (you need to launch the API `asr_server.py` first)
    AsrOpenaiWhisper - calls OpenAI Whisper API (fast??)
"""

import os, sys, time, argparse, re, logging, copy, pdb, msgpack
from termcolor import cprint
from dotenv import load_dotenv
from typing import Optional, List
from argparse import Namespace
from collections import Counter
import numpy as np
import torch, requests, json
from termcolor import cprint
from transformers import AutoTokenizer


load_dotenv("../.env", override=True)  # load API keys into
print(os.getenv("HF_HOME"))
sys.path.append("../")

import jax.numpy as jnp
from jax import jit
from whisper_jax import FlaxWhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from utils.utils import update_source_word_list, purge_directory, parse_language_pair
from chat_templates.templates import PROMPT_TEMPLATES


from vllm import LLM, SamplingParams

import io
from openai import OpenAI
from utils.utils import Timer

from scipy.io.wavfile import write as wav_write

# from configs import config_1 as CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=int, default=-1)
parser.add_argument("--model_id", type=str, default="")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--use_api", action="store_true")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--use_asr_api", action="store_true")
parser.add_argument("--asr_model_size", type=str, default="small")
parser.add_argument("--prompt_id", type=int, default=0)
parser.add_argument("--bgd_info", action="store_true")
parser.add_argument("--min_read_time", type=float, default=0)
parser.add_argument("--min_lag_words", type=int, default=1)
parser.add_argument("--func_wrds", type=str, default="[]")
parser.add_argument("--priming", action="store_true")
args, unknown_args = parser.parse_known_args()

from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction


# --------------------------------------------------------------------------------------------------------------------
purge_directory(args.output)
DEVICE = args.device
WAIT_K = args.k
verbose = args.verbose
use_asr_api = args.use_asr_api
PROMPT_ID = args.prompt_id
ENDPOINT = os.environ["VLLM_SERVER_ENDPOINT_URL"]
ASR_ENDPOINT = os.environ["ASR_SERVER_ENDPOINT_URL"]
print(ENDPOINT)
print(ASR_ENDPOINT)

ASR_MODEL_SIZE = args.asr_model_size

# FIXME: these should be CLI arguments
ASR_MODEL_NAME = f"openai/whisper-{ASR_MODEL_SIZE}.en"  # "openai/whisper-tiny.en" # "openai/whisper-small.en" "openai/whisper-large-v2"
if ASR_MODEL_SIZE == "large-v3":
    ASR_MODEL_NAME = "openai/whisper-large-v3"
if ASR_MODEL_SIZE == "distil-large-v3":
    ASR_MODEL_NAME = "distil-whisper/distil-large-v3"

SRATE = 16000
MIN_LAG_WORDS = int(args.min_lag_words)
MIN_READ_TIME = float(args.min_read_time)
RESPONSE_PRIMING = bool(args.priming)
cprint(f"RESPONSE_PRIMING: {RESPONSE_PRIMING}", "black", "on_light_magenta")

if args.model_id == "meta-llama/Llama-2-13b-chat-hf":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "microsoft/Orca-2-7b":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "meta-llama/Meta-Llama-3-70B-Instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "casperhansen/llama-3-8b-instruct-awq":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "casperhansen/llama-3-70b-instruct-awq":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "microsoft/Phi-3-mini-4k-instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["<|end|>", "<|endoftext|>"]
elif args.model_id == "google/gemma-7b-it":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["<end_of_turn>", "<eos>"]
elif args.model_id == "mistralai/Mistral-7B-Instruct-v0.2":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "mistralai/Mistral-7B-Instruct-v0.1":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
else:
    raise RuntimeError("Unknown model id")
model_id = args.model_id

# function_words = [
#     "the",
#     "a",
#     "is",
#     "am",
#     "in",
#     "out",
#     "by",
#     "on",
#     "off",
#     "down",
#     "up",
#     "off",
#     "and",
#     "will",
#     "to",
#     "from",
#     "not",
# ]
# function_words = ["the", "a", "is", "am", "to", "will", "not"]
# function_words = []

cprint(args.func_wrds, "red", "on_cyan")
function_words = args.func_wrds.split("_") if not args.func_wrds == "_" else []
cprint(f"function_words: {function_words}", "red", "on_cyan")
time.sleep(2)


# --------------------------------------------------------------------------------------------------------------------

SRC_LANG, TARG_LANG = parse_language_pair(args.dir)

A = []


def check_if_asr_model_is_right(asr_model_size: str) -> None:
    response = requests.post(f"{ASR_ENDPOINT}/info", json={})
    model_running_on_api = json.loads(response.text)["asr_model_name"]
    cprint(
        f"ASR model running at API: {model_running_on_api}",
        "black",
        "on_red",
        attrs=["bold"],
    )
    assert model_running_on_api == ASR_MODEL_NAME, "Wrong ASR model running at API."


# llm = LLM(model=model_id, max_num_seqs=1, max_model_len=4096, block_size=8)
if not args.use_api:
    tensor_parallel_size = 1  # torch.torch.cuda.device_count()
    llm = LLM(model=model_id, max_num_seqs=1, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    check_if_asr_model_is_right(ASR_MODEL_SIZE)


def get_last_word(outputs, prompt):
    return list(
        map(
            lambda x: " ".join(x["generated_text"][len(prompt) :].split(" ")[:-1]),
            outputs,
        )
    )


def build_full_prompt(partial_source, partial_target, background=None):

    DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATES[PROMPT_ID](SRC_LANG=SRC_LANG, TARG_LANG=TARG_LANG, BGD_INFO=background)

    if ACCEPTS_SYSTEM_MESSAGE:
        messages = [
            {"role": "system", "content": f"{DEFAULT_SYSTEM_PROMPT}"},
            {"role": "user", "content": f"context: {partial_source}"},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": f"{DEFAULT_SYSTEM_PROMPT}\nContext: {partial_source}",
            }
        ]

    tmp_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if RESPONSE_PRIMING:
        prompt = f"{tmp_prompt}{TARG_LANG} translation:{partial_target}"
    else:
        prompt = f"{tmp_prompt}{partial_target}"
    return re.sub(r"\s+([,.!?;:])", r"\1", prompt)


sampling_params = SamplingParams(
    temperature=0,
    min_tokens=2,
    max_tokens=20,
    stop=[" "],
    include_stop_str_in_output=True,
    # use_beam_search=True,
    # best_of=3,
    # top_p=0.95,
)


class Translator:

    def __init__(self, generation_kwargs, function_words):
        self.partial_target = []
        self.flags = dict(consecutive_duplicates=0)
        self.generation_kwargs = generation_kwargs
        self.FUNCTION_WORDS = function_words
        self.background = None

    def __generate(self, prompt, get_more_tokens=False):
        if args.use_api:
            prompt_len = len(prompt)
            payload = {"prompt": prompt}
            payload.update(
                {
                    k: v
                    for k, v in self.generation_kwargs.__dict__.items()
                    if k
                    in [
                        "temperature",
                        "min_tokens",
                        "max_tokens",
                        "stop",
                        "include_stop_str_in_output",
                        # "use_beam_search",
                        # "best_of",
                    ]
                }
            )

            # set min_tokens to 3 when requested (e.g. a single whitespace is generated)
            payload["min_tokens"] = 3 if get_more_tokens else self.generation_kwargs.min_tokens
            response = requests.post(ENDPOINT, json=payload)
            # pdb.set_trace()
            return json.loads(response.text)["text"][0][prompt_len:].replace("\n", "")

        else:
            result = llm.generate(prompts=[prompt], use_tqdm=False, sampling_params=self.generation_kwargs)
            return result[0].outputs[0].text.replace("\n", "")

    def _generate(self, prompt):

        just_generated = self.__generate(prompt)

        # NOTE: EXPERIMENTAL. Handles the edge case where the LLM doesn't prepend a space its output
        if not just_generated.endswith(" "):
            just_generated = f"{just_generated} "

        if just_generated == " ":  # NOTE: if only a space is generated, try again with more tokens
            just_generated = self.__generate(prompt, get_more_tokens=True)

        for eot_token_string in EOT_TOKEN_SEQUENCE:
            if eot_token_string in just_generated:
                self.flags["stop_reason"] = "eot_id"
                last_word = just_generated.split(eot_token_string)[0]
                self.flags["last_word"] = last_word
                return True

        # detect if <|start_header_id|> is generated, chop off what's after it (Llama-3 + ASR specific)
        if "<|start_header_id|>" in just_generated:
            self.flags["stop_reason"] = just_generated.split("<|start_header_id|>")[0]
            self.flags["stop_reason"] = "new_word"
            return True

        # detect if full stop, question or exclamation mark is generated, chop off what's after it
        just_generated, full_stop_generated = self._check_if_full_stop(just_generated)
        if full_stop_generated:
            self.flags["stop_reason"] = "new_word"
            self.flags["last_word"] = just_generated
            return True

        if " " in just_generated[1:]:
            self.flags["stop_reason"] = "new_word"
            self.flags["last_word"] = just_generated.split(" ")[-2]
            return True

        if self.flags["stop_reason"] is None:
            return True

        else:
            raise RuntimeError("Wrong stop reason.")

    def _step(self, partial_source, partial_target):
        self.flags["partial_target"] = partial_target
        prompt = build_full_prompt(partial_source, partial_target, background=self.background)

        if verbose:
            cprint(prompt, color="green")

        print("in the context: ", self.generation_kwargs.temperature)
        st = time.time()
        _ = self._generate(prompt)
        A.append(time.time() - st)

        print("outside the context: ", self.generation_kwargs.temperature)

        if verbose:
            cprint(f"stop_reason: {self.flags['stop_reason']}", color="magenta")

    def _check_if_full_stop(self, string):
        punctuation_marks = {".", "?", "!", ":"}
        for char in string:
            if char in punctuation_marks:
                string = string.split(char)[0]  # take the string before the FIRST punctuation mark
                if self.flags["source_finished"]:
                    return f"{string}{char}", True  # true EOS detected
                else:  # just insert a space before the punctuation mark, it will be dropped in _generate
                    return f"{string} {char}", False
        return (
            string,
            False,
        )  # if no punctuation mark is found, return the original string

    def _translate(self, partial_source_lst: List, source_finished: bool, k=4):

        self.flags["source_finished"] = source_finished
        self.flags["stop_reason"] = None
        self.flags["last_word"] = ""

        # if the new source word is a function word, READ action
        if (not source_finished) and (partial_source_lst[-1].lower() in self.FUNCTION_WORDS):
            return dict(action="READ", content="", final=False)

        # otherwise do the full cycle
        partial_source = " ".join(partial_source_lst)
        partial_source = re.sub(r"\s+([,.!?;:])", r"\1", partial_source)
        self.partial_target = [re.sub(r"\s+([,.!?;:])\s+", r"\1", it) for it in self.partial_target]

        partial_target = re.sub(r"\s+", " ", " ".join(self.partial_target))

        if verbose:
            with open(f"{args.output}/inc_asr.log", "a") as f:
                f.write(partial_source + "\n")
            with open(f"{args.output}/inc_trasnation.log", "a") as f:
                f.write(partial_target + "\n")

        self._step(partial_source, partial_target)

        if verbose:
            print(f"{partial_source}|{partial_target}|{self.flags['last_word']}|")
            print("-" * 89)

        # read more if the last genrated word is a duplicate of the previous generated word, otherwise append to target
        if len(self.partial_target) > 0:
            if self.partial_target[-1].lower() == self.flags["last_word"].lower():
                self.flags["consecutive_duplicates"] += 1
                if verbose:
                    print(f"{self.flags['consecutive_duplicates']} consecutive duplicates detected")
            else:
                self.flags["consecutive_duplicates"] = 0

        if self.flags["consecutive_duplicates"] > 2:
            return dict(action="READ", content="", final=False)

        # decide whether to append a new word (update the paratial_target)
        if self.flags["stop_reason"] in ["new_word", "eot_id"]:
            self.flags["last_word"], full_stop_detected = self._check_if_full_stop(self.flags["last_word"])
            if full_stop_detected and source_finished:
                self.partial_target.append(self.flags["last_word"])
                return dict(action="WRITE", content=self.flags["last_word"], final=True)
        if self.flags["stop_reason"] == "new_word":
            self.partial_target.append(self.flags["last_word"])
            return dict(action="WRITE", content=self.flags["last_word"], final=False)
        elif self.flags["stop_reason"] == "eot_id":
            if source_finished:
                self.partial_target.append(self.flags["last_word"])
                return dict(action="WRITE", content=self.flags["last_word"], final=True)
            else:
                return dict(action="WRITE", content="", final=False)
        else:
            # raise RuntimeError('asdfasd')
            cprint("******** INVALID STOP REASON *********", "black", "on_yellow")
            return dict(action="READ", content="", final=False)


class AsrJAX:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)

        self.srate = SRATE
        self.DEVICE = DEVICE

        if not use_asr_api:
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                ASR_MODEL_NAME, dtype=jnp.bfloat16, _do_init=True
            )

            self.p_generate = jit(self.generate_fn)
            self._warmup_asr()
        else:
            pass

    def _warmup_asr(self):
        cprint("Warming up the ASR model...", "black", "on_cyan", attrs=["bold"])
        input_features = self.processor(
            np.random.randn(32000), sampling_rate=self.srate, return_tensors="np"
        ).input_features
        self.p_generate(input_features)

    def generate_fn(self, input_features):
        pred_ids = self.model.generate(
            input_features,
            task="transcribe",
            return_timestamps=False,
            max_length=self.model.config.max_length,
            params=self.model.params,
        )
        return pred_ids.sequences

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:
        if not use_asr_api:
            input_features = self.processor(audio_array, sampling_rate=self.srate, return_tensors="np").input_features
            predicted_ids = self.p_generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        else:
            # NOTE: slow because we're sending float64 over JSON
            # response = requests.post(
            #     f"{ASR_ENDPOINT}/generate",
            #     json={"source": audio_array, "source_finished": source_finished}
            # )
            # NOTE: faster because we're sending float16 over msgpack
            response = requests.post(
                f"{ASR_ENDPOINT}/generate",
                data=msgpack.packb(
                    {
                        "source": np.array(audio_array).astype(np.float16).tolist(),
                        "source_finished": source_finished,
                    }
                ),
                headers={"Content-Type": "application/msgpack"},
            )
            transcription = json.loads(response.text)["recognized_word_list"]
        return self._postprocess(transcription, source_finished)


class Asr:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME, dtype=torch.float16)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_NAME,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.asr_model.config.forced_decoder_ids = None
        self.srate = SRATE
        self.DEVICE = DEVICE

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:
        input_features = self.processor(
            audio_array,
            sampling_rate=self.srate,
            return_tensors="pt",
            device=f"cuda:{self.DEVICE}",
            dtype=torch.float16,
            pad_to_multiple_of=128,
        ).input_features.to(device=f"cuda:{self.DEVICE}", dtype=torch.half)
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return self._postprocess(transcription, source_finished)


class AsrOpenaiWhisper:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)

        self.srate = SRATE
        self.DEVICE = DEVICE

        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY_MINE"))

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:

        cprint("Using OpenAI Whisper ASR", "black", "on_yellow", attrs=["bold"])
        buffer = io.BytesIO()
        buffer.name = "audio.wav"
        wav_write(buffer, SRATE, np.array(audio_array))
        buffer.seek(0)
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,  # Use the in-memory file-like object
            response_format="text",
            # prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
        )
        return self._postprocess(transcription, source_finished)


@entrypoint
class Agent(SpeechToTextAgent):

    def __init__(self, args: Optional[Namespace] = None) -> None:
        super().__init__(args)
        self.generation_kwargs = sampling_params
        self.function_words = function_words
        # self.asr_model = Asr(ASR_MODEL_NAME, DEVICE, SRATE)
        self.asr_model = AsrJAX(ASR_MODEL_NAME, DEVICE, SRATE)
        # self.asr_model = AsrOpenaiWhisper(ASR_MODEL_NAME, DEVICE, SRATE)
        self._reset()

    def _set_background(self, background):
        """the evaluator sets it when path to backgrounds is provided as a CLI argument"""
        self.translator.background = background

    def _save_asr_and_translation(self):
        with open(f"{args.output}/asr.log", "a") as f:
            f.write(" ".join(self.deduplicated_list_of_words) + "\n")
        with open(f"{args.output}/translation.log", "a") as f:
            f.write(" ".join(self.translator.partial_target) + "\n")

    def _reset(self):
        if verbose:
            cprint("resetting translator", color="red", attrs=["bold"])
        self.translator = Translator(function_words=self.function_words, generation_kwargs=self.generation_kwargs)
        self.deduplicated_list_of_words = []
        self.first_batch = True

    def policy(self):

        # NOTE: EXPERIMENTAL. Unconditionally return a READ action if the source is < 2.4 s
        if not self.states.source_finished:
            if len(self.states.source) / SRATE <= MIN_READ_TIME:
                return ReadAction()

        # FIXME: add docstring what does it do?
        recognized_word_list = self.asr_model.recognize(
            self.states.source,  # NOTE: we only take the last 3 seconds of audio
            self.states.source_finished,
        )

        # if no words are recognized yet (e.g. at the start of a sentence), return a READ action
        if len(recognized_word_list) == 0:
            return ReadAction()

        # add fresh input words to the existing list of input words, without duplicates
        _updated_source_word_list, num_words_added = update_source_word_list(
            self.deduplicated_list_of_words, recognized_word_list, verbose=verbose
        )

        # if no new words are added or even fewer than before, return a READ action
        if not self.states.source_finished:
            if len(_updated_source_word_list) <= len(self.deduplicated_list_of_words):
                cprint("No new words added", "white", "on_red", attrs=["bold"])
                return ReadAction()

        self.deduplicated_list_of_words = _updated_source_word_list

        # FIXME: experimental (ASR is only allowed to add one or zero words at a time)
        if not self.states.source_finished and not self.first_batch:
            lim = len(self.deduplicated_list_of_words) - max(1, num_words_added) + 1
            self.deduplicated_list_of_words = self.deduplicated_list_of_words[:lim]

        if (len(self.deduplicated_list_of_words) <= WAIT_K) and (not self.states.source_finished):
            return ReadAction()

        result = self.translator._translate(self.deduplicated_list_of_words, self.states.source_finished, k=WAIT_K)
        self.first_batch = False  # clear the first batch (of audio chunks) flag

        # cprint(self.deduplicated_list_of_words, color='cyan', attrs=['bold'])
        # if self.states.source_finished:
        #     return WriteAction('asdf', finished=True)
        # else:
        #     return WriteAction('asdf', finished=False)

        # if the source is finished, keep translating until the model outputs a final translation
        if self.states.source_finished:
            _result = ""
            while not result["final"]:
                result = self.translator._translate(
                    self.deduplicated_list_of_words,
                    self.states.source_finished,
                    k=WAIT_K,
                )
                _result = " ".join([_result, result["content"]])
            result["content"] = _result

        if verbose:
            cprint(f"{np.mean(A):.2f}, {np.std(A):.2f} N: {len(A)}", color="green")
            cprint(result, color="yellow")
            cprint(len(self.states.source), color="yellow")
            cprint(self.deduplicated_list_of_words, color="cyan", attrs=["bold"])

        if result["action"] == "READ":
            return ReadAction()
        elif result["action"] == "WRITE":
            if result["final"]:
                self._save_asr_and_translation()
                self._reset()
            return WriteAction(result["content"], finished=result["final"])
        else:
            raise RuntimeError("Unknown action")
