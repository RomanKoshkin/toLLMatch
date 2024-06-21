import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(__file__))
from s2t_beam_agent import S2TWithEDAttAgent
from spm_detok_agent import SentencePieceModelDetokenizerAgentForText

try:
    from simuleval.agents import AgentPipeline
    from simuleval.utils import entrypoint
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


@entrypoint
class S2TWithEDAttAgentPipeline(AgentPipeline):
    pipeline = [
        S2TWithEDAttAgent,
        SentencePieceModelDetokenizerAgentForText,
    ]

    def __init__(self, args) -> None:

        s2t = self.pipeline[0](args)
        detokenizer = self.pipeline[1](args)

        module_list = [s2t, detokenizer]

        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)
