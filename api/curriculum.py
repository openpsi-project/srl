from typing import Optional, List, Dict, Union
import logging

from base.names import curriculum_stage
from base.conditions import make as make_condition
import api.config
import base.name_resolve


class Curriculum:
    """Abstract class of a curriculum_learning. Curriculum controls the stages of training.
    Typically, evaluation manager will pass the evaluation result to curriculum_learning control and the curriculum_learning
    control will decide whether to change the global stage. As a result, changes will be made to the environments
    upon resetting.
    """

    def submit(self, data: Dict) -> bool:
        """Submit the episode info to the curriculum_learning.
        Args:
            data: episode info, typically results of a batch of evaluations.
        Returns:
            done (bool): whether the curriculum_learning is finished.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset the Curriculum.
        """
        raise NotImplementedError()

    def get_stage(self) -> str:
        """Get the current course of the curriculum.
        Returns:
            course_name(str): name of the current course.
        """
        raise NotImplementedError()


class LinearCurriculum(Curriculum):

    def __init__(self, experiment_name, trial_name, curriculum_name, stages: Union[str, List[str]],
                 condition_cfg: List[api.config.Condition]):
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__curriculum_name = curriculum_name
        self.logger = logging.getLogger(f"Curriculum {self.__curriculum_name}")
        self.__conditions = [make_condition(cond) for cond in condition_cfg]
        if isinstance(stages, str):
            self.__stages = [stages]
        else:
            self.__stages = stages
        self.__stage_index = 0

    def reset(self):
        self.__stage_index = 0
        self.set_stage(self.__stages[self.__stage_index])

    def set_stage(self, stage):
        self.logger.info(f"now on stage {stage}")
        base.name_resolve.add(curriculum_stage(self.__experiment_name, self.__trial_name,
                                               self.__curriculum_name),
                              value=stage,
                              replace=True)

    def submit(self, data):
        for cond in self.__conditions:
            if not cond.is_met_with(data):
                self.logger.info(f"Condition {cond} is not met.")
                return False
        else:
            self.logger.info("All conditions met.")
            if self.__stage_index + 1 == len(self.__stages):
                self.logger.info(f"All stages cleared: {self.__stages}")
                return True
            else:
                self.__stage_index += 1
                self.set_stage(self.__stages[self.__stage_index])
                return False

    def get_stage(self) -> Optional[str]:
        try:
            return base.name_resolve.get(
                curriculum_stage(self.__experiment_name, self.__trial_name, self.__curriculum_name))
        except base.name_resolve.NameEntryNotFoundError:
            return None


def make(cfg: api.config.Curriculum, worker_info: api.config.WorkerInformation):
    if cfg.type_ == api.config.Curriculum.Type.Linear:
        return LinearCurriculum(
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name,
            curriculum_name=cfg.name,
            stages=cfg.stages,
            condition_cfg=cfg.conditions,
        )
    else:
        raise NotImplementedError()
