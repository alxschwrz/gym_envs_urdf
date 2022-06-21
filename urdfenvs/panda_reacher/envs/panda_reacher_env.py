import numpy as np
from urdfenvs.panda_reacher.resources.panda_robot import PandaRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv
from forwardkinematics.urdfFks.pandaFk import PandaFk


class PandaReacherEnv(UrdfEnv):
    def __init__(self, friction=0.0, gripper=False, **kwargs):
        super().__init__(
            PandaRobot(gripper=gripper, friction=friction), **kwargs
        )
        self._fk = PandaFk()
        self.set_spaces()
        self._goalEnv = True  # OpenAI GoalEnv with dict observations: observation, achieved_goal, desired_goal
        if self._goalEnv:
            ob = self.reset()
            self.convert_observation_space_to_goalEnv(ob['observation'].shape, tuple((self._goals[0].m(),)))


    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            pos = np.zeros(self._robot.n())
            pos[3] = -1.501
            pos[5] = 1.8675
            pos[6] = np.pi / 4
            if self._robot.n() > 7:
                pos[7] = 0.02
                pos[8] = 0.02
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel

    def get_ee_position(self):
        joint_states = self._robot.get_observation()['joint_state']['position']
        ee_position = self._fk.fk(joint_states, 7, positionOnly=True)
        return ee_position

    def convert_observation_space_to_goalEnv(self, observation_shape: tuple, goal_shape: tuple):
        # todo: limits are hardcoded
        import gym
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=goal_shape, dtype=np.float32),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=goal_shape, dtype=np.float32),
            )
        )
