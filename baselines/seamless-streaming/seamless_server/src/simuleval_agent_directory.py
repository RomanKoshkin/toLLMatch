# Creates a directory in which to look up available agents

import os
from typing import List, Optional
from src.simuleval_transcoder import SimulevalTranscoder
import json
import logging

logger = logging.getLogger("socketio_server_pubsub")

# fmt: off
M4T_P0_LANGS = [
    "eng",
    "arb", "ben", "cat", "ces", "cmn", "cym", "dan",
    "deu", "est", "fin", "fra", "hin", "ind", "ita",
    "jpn", "kor", "mlt", "nld", "pes", "pol", "por",
    "ron", "rus", "slk", "spa", "swe", "swh", "tel",
    "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie",
]
# fmt: on


class NoAvailableAgentException(Exception):
    pass


class AgentWithInfo:
    def __init__(
        self,
        agent,
        name: str,
        modalities: List[str],
        target_langs: List[str],
        # Supported dynamic params are defined in StreamingTypes.ts
        dynamic_params: List[str] = [],
        description="",
        has_expressive: Optional[bool] = None,
    ):
        self.agent = agent
        self.has_expressive = has_expressive
        self.name = name
        self.description = description
        self.modalities = modalities
        self.target_langs = target_langs
        self.dynamic_params = dynamic_params

    def get_capabilities_for_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "modalities": self.modalities,
            "targetLangs": self.target_langs,
            "dynamicParams": self.dynamic_params,
        }

    @classmethod
    def load_from_json(cls, config: str):
        """
        Takes in JSON array of models to load in, e.g.
        [{"name": "s2s_m4t_emma-unity2_multidomain_v0.1", "description": "M4T model that supports simultaneous S2S and S2T", "modalities": ["s2t", "s2s"], "targetLangs": ["en"]},
        {"name": "s2s_m4t_expr-emma_v0.1", "description": "ES-EN expressive model that supports S2S and S2T", "modalities": ["s2t", "s2s"], "targetLangs": ["en"]}]
        """
        configs = json.loads(config)
        agents = []
        for config in configs:
            agent = SimulevalTranscoder.build_agent(config["name"])
            agents.append(
                AgentWithInfo(
                    agent=agent,
                    name=config["name"],
                    modalities=config["modalities"],
                    target_langs=config["targetLangs"],
                )
            )
        return agents


class SimulevalAgentDirectory:
    # Available models. These are the directories where the models can be found, and also serve as an ID for the model.
    seamless_streaming_agent = "SeamlessStreaming"
    seamless_agent = "Seamless"

    def __init__(self):
        self.agents = []
        self.did_build_and_add_agents = False

    def add_agent(self, agent: AgentWithInfo):
        self.agents.append(agent)

    def build_agent_if_available(self, model_id, config_name=None):
        agent = None
        try:
            if config_name is not None:
                agent = SimulevalTranscoder.build_agent(
                    model_id,
                    config_name=config_name,
                )
            else:
                agent = SimulevalTranscoder.build_agent(
                    model_id,
                )
        except Exception as e:
            from fairseq2.assets.error import AssetError
            logger.warning("Failed to build agent %s: %s" % (model_id, e))
            if isinstance(e, AssetError):
                logger.warning(
                    "Please download gated assets and set `gated_model_dir` in the config"
                )
            raise e

        return agent

    def build_and_add_agents(self, models_override=None):
        if self.did_build_and_add_agents:
            return

        if models_override is not None:
            agent_infos = AgentWithInfo.load_from_json(models_override)
            for agent_info in agent_infos:
                self.add_agent(agent_info)
        else:
            s2s_agent = None
            if os.environ.get("USE_EXPRESSIVE_MODEL", "0") == "1":
                logger.info("Building expressive model...")
                s2s_agent = self.build_agent_if_available(
                    SimulevalAgentDirectory.seamless_agent,
                    config_name="vad_s2st_sc_24khz_main.yaml",
                )
                has_expressive = True
            else:
                logger.info("Building non-expressive model...")
                s2s_agent = self.build_agent_if_available(
                    SimulevalAgentDirectory.seamless_streaming_agent,
                    config_name="vad_s2st_sc_main.yaml",
                )
                has_expressive = False

            if s2s_agent:
                self.add_agent(
                    AgentWithInfo(
                        agent=s2s_agent,
                        name=SimulevalAgentDirectory.seamless_streaming_agent,
                        modalities=["s2t", "s2s"],
                        target_langs=M4T_P0_LANGS,
                        dynamic_params=["expressive"],
                        description="multilingual expressive model that supports S2S and S2T",
                        has_expressive=has_expressive,
                    )
                )

        if len(self.agents) == 0:
            logger.error(
                "No agents were loaded. This likely means you are missing the actual model files specified in simuleval_agent_directory."
            )

        self.did_build_and_add_agents = True

    def get_agent(self, name):
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_agent_or_throw(self, name):
        agent = self.get_agent(name)
        if agent is None:
            raise NoAvailableAgentException("No agent found with name= %s" % (name))
        return agent

    def get_agents_capabilities_list_for_json(self):
        return [agent.get_capabilities_for_json() for agent in self.agents]
