import os, sys, time, argparse, re, logging, copy, pdb
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

from utils.utils import purge_directory, parse_language_pair
from vllm import LLM, SamplingParams
from chat_templates.templates import PROMPT_TEMPLATES

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
parser.add_argument("--prompt_id", type=int, default=0)
parser.add_argument("--func_wrds", type=str, default="")
parser.add_argument("--priming", action="store_true")
args, unknown_args = parser.parse_known_args()

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction


# --------------------------------------------------------------------------------------------------------------------
purge_directory(args.output)
DEVICE = args.device
WAIT_K = args.k
verbose = args.verbose
PROMPT_ID = args.prompt_id
ENDPOINT = os.environ["VLLM_SERVER_ENDPOINT_URL"]
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

# function_words = ['the', 'a', 'is', 'am', 'in', 'out', 'by', 'on', 'off', 'down', 'up', 'off', 'and', 'will', 'to', 'from', 'not']
# function_words = ["the", "a", "is", "am", "to", "will", "not"]
function_words = args.func_wrds.split("_") if not args.func_wrds == "_" else []
cprint(f"function_words: {function_words}", "red", "on_cyan")
time.sleep(2)


# --------------------------------------------------------------------------------------------------------------------

SRC_LANG, TARG_LANG = parse_language_pair(args.dir)

A = []


# llm = LLM(model=model_id, max_num_seqs=1, max_model_len=4096, block_size=8)
if not args.use_api:
    tensor_parallel_size = 1  # torch.torch.cuda.device_count()
    llm = LLM(model=model_id, max_num_seqs=1, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)


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

        tmp_str, full_stop_generated = self._check_if_full_stop(just_generated)
        if full_stop_generated:
            self.flags["stop_reason"] = "new_word"
            self.flags["last_word"] = tmp_str
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
        outputs = self._generate(prompt)
        A.append(time.time() - st)

        print("outside the context: ", self.generation_kwargs.temperature)

        if verbose:
            cprint(f"stop_reason: {self.flags['stop_reason']}", color="magenta")

    def _check_if_full_stop(self, string):
        punctuation_marks = {".", "?", "!"}
        for char in string:
            if char in punctuation_marks:
                string = string.split(char)[0]  # take the string before the FIRST punctuation mark
                if self.flags["source_finished"]:
                    return f"{string}{char}", True  # true EOS detected
                else:  # just drop the punctuation mark to enable the model to continue
                    return string, False
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
            with open(f"{args.output}/inc_trasnation.log", "a") as f:
                f.write(partial_target + "\n")
            with open(f"{args.output}/inc_src.log", "a") as f:
                f.write(partial_source + "\n")

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

        # decide whether to append a new word
        if self.flags["stop_reason"] in ["new_word", "eot_id"]:
            self.flags["last_word"], full_stop_detected = self._check_if_full_stop(self.flags["last_word"])
            if full_stop_detected and source_finished:
                self.partial_target.append(self.flags["last_word"])
                return dict(action="WRITE", content=self.flags["last_word"], final=True)
        if self.flags["stop_reason"] == "new_word":
            self.partial_target.append(self.flags["last_word"])
            # cprint(self.partial_target, color='red') # NOTE: DEBUG
            return dict(action="WRITE", content=self.flags["last_word"], final=False)
        elif self.flags["stop_reason"] == "eot_id":
            if source_finished:
                self.partial_target.append(self.flags["last_word"])
                return dict(action="WRITE", content=self.flags["last_word"], final=True)
            else:
                return dict(action="WRITE", content="", final=False)
        else:
            # raise RuntimeError('asdfasd')
            cprint(f"******** INVALID STOP REASON *********", "black", "on_yellow")
            return dict(action="READ", content="", final=False)


@entrypoint
class Agent(TextToTextAgent):

    def __init__(self, args: Optional[Namespace] = None) -> None:
        super().__init__(args)
        self.generation_kwargs = sampling_params
        self.function_words = function_words
        self._reset()
        try:
            os.remove(f"{args.output}/translation")
        except:
            pass

    def _set_background(self, background):
        """the evaluator sets it when path to backgrounds is provided as a CLI argument"""
        self.translator.background = background

    def _save_asr_and_translation(self):
        with open(f"{args.output}/translation", "a") as f:
            f.write(" ".join(self.translator.partial_target) + "\n")

    def _reset(self):
        if verbose:
            cprint("resetting translator", color="red", attrs=["bold"])
        self.translator = Translator(function_words=self.function_words, generation_kwargs=self.generation_kwargs)

    def policy(self):

        if (len(self.states.source) <= WAIT_K) and (not self.states.source_finished):
            return ReadAction()

        result = self.translator._translate(self.states.source, self.states.source_finished, k=WAIT_K)
        # time.sleep(0.5) # NOTE: DEBUG
        if verbose:
            cprint(f"{np.mean(A):.2f}, {np.std(A):.2f} N: {len(A)}", color="green")
            cprint(result, color="yellow")
        if result["action"] == "READ":
            return ReadAction()
        elif result["action"] == "WRITE":
            if result["final"]:
                self._save_asr_and_translation()
                self._reset()
            return WriteAction(result["content"], finished=result["final"])
        else:
            raise RuntimeError("Unknown action")
