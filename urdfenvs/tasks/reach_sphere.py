from typing import Any, Dict, Union, List
import numpy as np


class ReachSphereTask(object):
    def __init__(self, reward_type: str = "sparse"):
        self._task_id = np.array([1])
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

    def sampleGoal(self, goal_limits):
        from MotionPlanningGoal.staticSubGoal import StaticSubGoal
        goal_limits = goal_limits['pos']
        desired_position = [np.random.uniform(goal_limits['x'][0], goal_limits['x'][1]),
                            np.random.uniform(goal_limits['y'][0], goal_limits['y'][1]),
                            np.random.uniform(goal_limits['z'][0], goal_limits['z'][1])]
        #desired_position = [0.3, -0.2, 0.5]
        goalDict = {"m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
                     'desired_position': desired_position, 'epsilon': 0.1, 'type': "staticSubGoal"}
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

    def get_achieved_goal(self, robot, fk):
        joint_states = robot.get_observation()['joint_state']['position']
        achieved_goal = fk.fk(joint_states, -1, positionOnly=True)
        return achieved_goal
