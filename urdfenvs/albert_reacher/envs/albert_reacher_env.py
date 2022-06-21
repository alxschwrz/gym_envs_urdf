import numpy as np

from urdfenvs.albert_reacher.resources.albert_robot import AlbertRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv
from forwardkinematics.urdfFks.albertFk import AlbertFk


class AlbertReacherEnv(UrdfEnv):
    """Albert reacher environment."""

    def __init__(self, **kwargs):
        super().__init__(robot=AlbertRobot(), task_list=["albert"], **kwargs)
        self._goalEnv = True
        self._fk = AlbertFk()
        self.set_spaces()
        if self._goalEnv:
            ob = self.reset()
            self.convert_observation_space_to_goalEnv(ob['observation'].shape, tuple((self._goals[0].m(),)))

    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n()+1:
            pos = np.zeros(self._robot.n()+1)
            pos[6] = -1.501
            pos[8] = 1.8675
            pos[9] = np.pi/4
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel

    def get_ee_position(self):
        joint_states = self._robot.get_observation()['joint_state']['position']
        ee_position = self._fk.fk(joint_states, "panda_link9", positionOnly=True)
        return ee_position
