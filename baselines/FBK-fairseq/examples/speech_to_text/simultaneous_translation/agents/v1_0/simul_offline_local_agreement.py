# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import BaseSimulSTAgent
from examples.speech_to_text.simultaneous_translation.agents.v1_0.base_simulst_agent import FairseqSimulSTAgent

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class LocalAgreementSimulSTAgent(FairseqSimulSTAgent):
    """
    Local Agreement agent for Simultaneous Speech Translation based on
    "Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection"
    (https://www.isca-speech.org/archive/pdfs/interspeech_2020/liu20s_interspeech.pdf)
    by Liu et al., 2020. The agent displays the agreeing prefixes of the two consecutive chunks:
    during the first n−1 chunks, no output is produced; from the n-th chunk on, the longest common prefix
    of the n consecutive chunks is identified and emitted.
    Empirically, the authors found that n=2 works better.
    The implementation works only for SentencePiece up to now.
    """
    def __init__(self, args):
        super().__init__(args)
        # Local Agreement using last 2 generated sentences as memory
        self.la_n = 2

    @staticmethod
    def add_args(parser):
        # fmt: off
        BaseSimulSTAgent.add_args(parser)
        # fmt: on
        return parser

    def initialize_states(self, states):
        super().initialize_states(states)
        states.chunks_hyp = []

    def update_states_read(self, states):
        super().update_states_read(states)
        if not states.finish_read():
            states.new_segment = True

    def common_prefix(self, states, prefix_tokens):
        """
        This method takes *states* as input, which stores the hypothesis generated at
        each time step in *states.chunks_hyp*, and returns the common prefix among the
        last *self.la_n* hypotheses without including the already emitted prefix.
        """
        if len(states.chunks_hyp) < self.la_n:
            return []

        prefix_len = self._get_prefix_len(prefix_tokens)
        candidates = [s[prefix_len:] for s in states.chunks_hyp[-self.la_n:]]
        common_pref = []
        for candidate in zip(*candidates):
            prefix_candidate = candidate[0]
            if all(prefix_el == prefix_candidate for prefix_el in candidate) and prefix_candidate != self.eos_idx:
                common_pref.append(prefix_candidate)
            else:
                break

        return common_pref

    def _emit_remaining_tokens(self, states):
        prefix_tokens = self._get_prefix(states)
        states.write = states.chunks_hyp[-1][self._get_prefix_len(prefix_tokens):]

    def _policy(self, states):
        """
        It generates a translation hypothesis starting from the encoder states
        contained in *states.encoder_states*, and applies the *self.prefix()*
        method to obtain the common prefix among the previously generated
        hypotheses. It returns READ_ACTION if the prefix is empty, meaning that
        there is no common prefix among the generated hypotheses, and
        WRITE_ACTION otherwise.
        """
        states.new_segment = False
        prefix_tokens = self._get_prefix(states)
        hypo = self.generate_hypothesis(states, prefix_tokens)
        hypo_tokens = hypo['tokens'].int()

        states.chunks_hyp.append(hypo_tokens)
        common_pref = self.common_prefix(states, prefix_tokens)

        if len(common_pref) > 0:
            states.write = common_pref
            return WRITE_ACTION
        return READ_ACTION
