from typing import Any, Dict, Union, List
import numpy as np


class FullTask(object):
    def __init__(self, reward_type: str = "sparse"):
        self._task_id = np.array([0])
        self.reward_type = reward_type

    def task_id(self) -> np.ndarray:
        return self._task_id

    def adapt_goal_to_task(self, goal):
        return goal

    def compute_reward(self, achieved_goal, desired_goal, goals):
        # dist = np.linalg.norm(achieved_goal[:3] - desired_goal[:3], axis=-1)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -np.array(dist > goals[0].epsilon() * 2, dtype=np.float32)
        else:
            return -dist

    def sampleGoal(self, goal_limits):
        from MotionPlanningGoal.staticSubGoal import StaticSubGoal
        desired_position = [np.random.uniform(goal_limits['pos']['x'][0], goal_limits['pos']['x'][1]),
                            np.random.uniform(goal_limits['pos']['y'][0], goal_limits['pos']['y'][1]),
                            np.random.uniform(goal_limits['pos']['z'][0], goal_limits['pos']['z'][1])
                            ]
        desired_direction = [np.random.uniform(goal_limits['ori']['x'][0], goal_limits['ori']['x'][1]),
                             np.random.uniform(goal_limits['ori']['y'][0], goal_limits['ori']['y'][1]),
                             np.random.uniform(goal_limits['ori']['z'][0], goal_limits['ori']['z'][1])
                             ]

        #desired_position = [0.3, 0.3, 0.4]
        #desired_direction = [np.random.uniform(0, 1.0), np.random.uniform(-1, 1), np.random.uniform(-1, 0)]

        # todo: the goal type will be a problem!
        goalDict = {"m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
                    'desired_position': desired_position, 'angle': desired_direction, 'epsilon': 0.1, 'type': "staticSubGoal"}
        goal = StaticSubGoal(name="goal", contentDict=goalDict)
        return goal

    def _compute_success(self, achieved_goal, desired_goal, done, goals):
        """
        Task-specific success-function
        """
        if done:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return bool(dist < goals[0].epsilon() * 2)
        return False

    def get_achieved_goal(self, robot, fk):
        joint_states = robot.get_observation()['joint_state']['position']
        achieved_position = fk.fk(joint_states, -1, positionOnly=True)
        x0 = fk.fk(joint_states, -2, positionOnly=True)

        direction = achieved_position - x0
        normalized_direction = direction / np.sqrt(np.sum(direction ** 2))
        achieved_direction = normalized_direction

        achieved_goal = np.zeros(6)
        achieved_goal[:3] = achieved_position
        achieved_goal[3:] = achieved_direction
        return achieved_goal

    def get_desired_goal(self, goals):
        desired_goal = np.zeros(6)
        desired_goal[:3] = goals[0].position()
        desired_goal[3:] = goals[0].angle()
        return desired_goal
