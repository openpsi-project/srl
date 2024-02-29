from typing import Dict, Union

from api.trainer import SampleBatch
import api.config as config


class PopulationAlgorithm:

    def configure(self, actors, policies, trainers, eval_managers):
        """Passes initial experiment configuration of different workers.
        """
        raise NotImplementedError()

    def step(self, sample: SampleBatch) -> Dict[str, Dict]:
        """Updates experiment configuration and returns requests to send to workers.
        Args:
            sample: sample received by population sample stream.
        Returns:
            requests: dictionary of requests to be send, where keys are command-s, values are kwargs dict-s. 
              This makes it easier for population_manager to call  group_request().

            E.g., for vanilla_pbt, `requests` should be like
            requests = {
                "pause": {"worker_names": ["policy/0", "trainer/0"]},
                "reconfigure": {
                    "worker_names": ["trainer/0"],
                    "worker_kwargs": [dict(config=trainer_config)]
                },
                "start": {"worker_names": ["policy/0", "trainer/0"]},
            }
        """
        raise NotImplementedError()


ALL_PBT_CLASSES = {}


def register(name, pbt_class):
    ALL_PBT_CLASSES[name] = pbt_class


def make(cfg: Union[str, config.PopulationAlgorithm]) -> PopulationAlgorithm:
    if isinstance(cfg, str):
        cfg = config.PopulationAlgorithm(type_=cfg)
    cls = ALL_PBT_CLASSES[cfg.type_]
    return cls(**cfg.args)
