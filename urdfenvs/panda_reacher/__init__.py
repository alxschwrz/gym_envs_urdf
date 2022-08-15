from gym.envs.registration import register
register(
    id='panda-reacher-tor-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherTorEnv'
)
register(
    id='panda-reacher-vel-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherVelEnv',
    max_episode_steps=1000  # for smoother Value-function; based on panda_gym example
)
register(
    id='panda-reacher-acc-v0',
    entry_point='urdfenvs.panda_reacher.envs:PandaReacherAccEnv'
)
