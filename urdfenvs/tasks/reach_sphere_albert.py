from typing import Any, Dict, Union, List
import numpy as np


class AlbertReachSphereTask(object):
    def __init__(self, reward_type: str = "sparse"):
        self._task_id = np.array([0])
        self.reward_type = reward_type

    def task_id(self) -> np.ndarray:
        return self._task_id

    def adapt_goal_to_task(self, goal):
        return goal

    def compute_reward(self, achieved_goal, desired_goal, goals):
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -np.array(dist > goals[0].epsilon(), dtype=np.float32)
        else:
            return -dist

    def sampleGoal(self):
        from MotionPlanningGoal.staticSubGoal import StaticSubGoal
        goalDict = {"m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
                    'desired_position': [np.random.uniform(0.0, 1.0),
                                         np.random.uniform(-1.0, 1.0),
                                         np.random.uniform(0.7, 1.5)
                                         ], 'epsilon': 0.1, 'type': "staticSubGoal"}
        goal = StaticSubGoal(name="goal", contentDict=goalDict)
        return goal

    def _compute_success(self, achieved_goal, desired_goal, done, goals):
        '''
        Task-specific success-function
        '''
        if done:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return bool(dist < goals[0].epsilon())
        return False
