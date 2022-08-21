import numpy as np
import itertools
import d4rl

from collections import defaultdict
from .base import GymEnv

from pytorch_lightning.utilities.parsing import AttributeDict as AttrDict


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""

    SUBTASKS = [
        "microwave",
        "kettle",
        "slide cabinet",
        "hinge cabinet",
        "bottom burner",
        "light switch",
        "top burner",
    ]

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return (
            obs,
            np.float64(rew),
            done,
            self._postprocess_info(info),
        )  # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    @staticmethod
    def subgoal_distance(obs, subgoal):
        """Computes L2 distance between subgoal and current state for subgoal reaching reward."""
        return np.linalg.norm(subgoal - obs[: subgoal.shape[0]])

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info["completed_tasks"]
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = (
                1 if task in completed_subtasks or self.solved_subtasks[task] else 0
            )
        return info
