from typing import Any, Dict, Union, List
import numpy as np


class PointTask(object):
    def __init__(self, reward_type: str = "sparse"):
        self._task_id = np.array([2])
        self.reward_type = reward_type

    def task_id(self) -> np.ndarray:
        return self._task_id

    def adapt_goal_to_task(self, goal):
        return goal

    def compute_reward(self, achieved_goal, desired_goal, goals):
        #dist2 = np.linalg.norm(achieved_goal[3:] - desired_goal[3:], axis=-1)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -np.array(dist > goals[0].epsilon(), dtype=np.float32)
        else:
            return -dist

    def sampleGoal(self, goal_limits):
        goal_limits = goal_limits['ori']
        desired_direction = np.array([np.random.uniform(goal_limits['x'][0], goal_limits['x'][1]),
                                      np.random.uniform(goal_limits['y'][0], goal_limits['y'][1]),
                                      np.random.uniform(goal_limits['z'][0], goal_limits['z'][1])])
        # desired_direction = np.random.uniform(-1.0, 1.0, 3)
        normalized_goal_direction = desired_direction / np.sqrt(np.sum(desired_direction ** 2))
        from MotionPlanningGoal.staticSubGoal import StaticSubGoal
        goalDict = {"m": 3, "w": 1.0, "prime": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
                    'desired_position': normalized_goal_direction, 'epsilon': 0.15, 'type': "staticSubGoal"}

        goal = StaticSubGoal(name="goal", contentDict=goalDict)
        return goal

    def _compute_success(self, achieved_goal, desired_goal, done, goals):
        '''
        Task-specific success-function
        '''
        if done:
            dist = np.linalg.norm(achieved_goal[3:] - desired_goal[3:], axis=-1)
            return bool(dist < goals[0].epsilon())
        return False

    def get_achieved_goal(self, robot, fk):
        joint_states = robot.get_observation()['joint_state']['position']
        x0 = fk.fk(joint_states, -2, positionOnly=True)
        ee = fk.fk(joint_states, -1, positionOnly=True)
        # l0 = fk.fk(joint_states, 0, positionOnly=True)
        # l1 = fk.fk(joint_states, 1, positionOnly=True)
        # l2 = fk.fk(joint_states, 2, positionOnly=True)
        # l3 = fk.fk(joint_states, 3, positionOnly=True)
        # l4 = fk.fk(joint_states, 4, positionOnly=True)
        direction = ee - x0
        normalized_direction = direction / np.sqrt(np.sum(direction ** 2))
        #achieved_goal = normalized_direction
        achieved_goal = np.zeros(6)
        achieved_goal[3:] = normalized_direction

        '''
        import pybullet as p
        import math
        duration = 0.1
        p.addUserDebugLine(lineFromXYZ=np.zeros(3),
                           lineToXYZ=normalized_direction,
                           lineColorRGB=(0, 128, 0),
                           lineWidth=10,
                           lifeTime=duration)
        p.addUserDebugLine(lineFromXYZ=ee,
                           lineToXYZ=ee+normalized_direction,
                           lineColorRGB=(0, 128, 0),
                           lineWidth=10,
                           lifeTime=duration)
                           
        '''

        '''
        import pybullet as p
        import math
        duration = 0.1
        p.addUserDebugLine(lineFromXYZ=ee,
                           lineToXYZ=(ee + (ee - x0)),
                           lineColorRGB=(0, 128, 0),
                           lineWidth=10,
                           lifeTime=duration)

        p.addUserDebugLine(lineFromXYZ=l0,
                           lineToXYZ=[0.5, 1, 0],
                           lineColorRGB=(255, 0, 0),
                           lineWidth=100,
                           lifeTime=duration)

        p.addUserDebugLine(lineFromXYZ=l0,
                           lineToXYZ=[-0.5, 1, 0],
                           lineColorRGB=(255, 0, 0),
                           lineWidth=100,
                           lifeTime=duration)
        shift = [0.2, 0, 0.1]
        pitch = np.rad2deg(math.asin(normalized_v[1])) / 100
        yaw = np.rad2deg(math.atan2(normalized_v[0], normalized_v[2])) / 100
        p.addUserDebugLine(lineFromXYZ=l0,
                           lineToXYZ=normalized_v,
                           lineColorRGB=(256, 0, 0),
                           lineWidth=10,
                           lifeTime=duration)

        import random
        for _ in range(10):
            debug_line = [0.5 + random.uniform(-0.1, 0.1),
                          0.5 + random.uniform(-0.1, 0.1),
                          0 + random.uniform(-0.1, 0.1)]

            p.addUserDebugLine(lineFromXYZ=l0,
                               lineToXYZ=debug_line,
                               lineColorRGB=(0, 0, 0),
                               lineWidth=10,
                               lifeTime=duration)
       '''
        return achieved_goal


    def get_desired_goal(self, goals):
        desired_goal = np.zeros(6)
        desired_goal[3:] = goals[0].position()
        return desired_goal
