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
import unittest
from unittest.mock import patch

from examples.speech_to_text.simultaneous_translation.agents.v1_0.simul_offline_local_agreement import \
    LocalAgreementSimulSTAgent
from fbk_simul_uts.v1_0.test_base_simulst_agent import BaseSTAgentTestCase


class LocalAgreementSimulSTPolicyTestCase(BaseSTAgentTestCase, unittest.TestCase):
    def create_agent(self):
        return LocalAgreementSimulSTAgent(self.args)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_local_agreement.LocalAgreementSimulSTAgent.load_model_vocab')
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'base_simulst_agent.FairseqSimulSTAgent.__init__')
    def setUp(self, mock_load_model_vocab, mock_simulst_agent_init):
        mock_simulst_agent_init.return_value = None
        mock_load_model_vocab.return_value = None
        self.base_init()

    def test_incomplete_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["I", "am", "an", "elephant."]]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, ["I", "am"])

    def test_complete_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, ["I", "am", "a", "quokka."])

    def test_empty_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["Hello", "I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, [])

    def test_empty_chunks(self):
        self.states.chunks_hyp = []
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, [])

    def test_one_chunk(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, [])

    def test_prefix(self):
        self.states.chunks_hyp = [
            ["I", "am", "a", "quokka."],
            ["I", "am", "an", "elephant."]
        ]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, [["I"]])
        self.assertEqual(prefix, ["am"])

    def test_three_chunks(self):
        self.states.chunks_hyp = [
            ["Hello", "I", "am", "a", "quokka."],
            ["I", "am", "a", "quokka."],
            ["I", "am", "an", "elephant."]
        ]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, ["I", "am"])

    def test_multilang_prefix(self):
        # Tags must be present in the sentences and removed only during the
        # write operation.
        self.states.chunks_hyp = [
            ["<langtag>", "I", "am", "a", "quokka."],
            ["<langtag>", "I", "am", "an", "elephant."]
        ]
        prefix = LocalAgreementSimulSTAgent.common_prefix(self.agent, self.states, None)
        self.assertEqual(prefix, ["<langtag>", "I", "am"])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_local_agreement.LocalAgreementSimulSTAgent._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_values = None
        super().test_finish_read()


if __name__ == '__main__':
    unittest.main()
