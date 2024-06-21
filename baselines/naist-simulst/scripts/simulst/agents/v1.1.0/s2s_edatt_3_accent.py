import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(__file__))
from s2t_beam_agent import S2TWithEDAttAgent
from tts_agent_3_accent import Subword2Yomi2SpeechAgent

try:
    from simuleval.agents import AgentPipeline
    from simuleval.utils import entrypoint
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


@entrypoint
class S2SEDAttWithAccentTTSAgentPipeline(AgentPipeline):
    pipeline = [
        S2TWithEDAttAgent,
        Subword2Yomi2SpeechAgent,
    ]

    def __init__(self, args) -> None:

        s2t = self.pipeline[0](args)
        tts = self.pipeline[1](args)

        module_list = [s2t, tts]

        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)
