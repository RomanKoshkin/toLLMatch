import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(__file__))
from t2t_beam_agent import T2TBeamLocalAgreementAgent
from spm_detok_agent import SentencePieceModelDetokenizerAgent
from text_filter_agent import LongTextFilterAgent

try:
    from simuleval.agents import AgentPipeline
    from simuleval.utils import entrypoint
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


@entrypoint
class T2TLocalAgreementAgentPipeline(AgentPipeline):
    pipeline = [
        T2TBeamLocalAgreementAgent,
        LongTextFilterAgent,
    ]

    def __init__(self, args) -> None:

        t2t = self.pipeline[0](args)
        detokenizer = self.pipeline[1](args)

        module_list = [t2t, detokenizer]

        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)
