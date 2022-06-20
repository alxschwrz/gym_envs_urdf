import gym
import pandas as pd

import urdfenvs.albert_reacher
import numpy as np
from stable_baselines3.sac import SAC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import time


def main():
    gripper = False
    env = gym.make("albert-reacher-vel-v0", dt=0.01, render=False)
    eval_env = gym.make("albert-reacher-vel-v0", dt=0.01, render=False)
    check_env(env)
    env = Monitor(env)
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(env, eval_freq=10_000, n_eval_episodes=5,
                                 best_model_save_path='models/debug/training/albert/',
                                 deterministic=True, render=False)

    check_for_correct_spaces(env, env.observation_space, env.action_space)
    model = SAC('MultiInputPolicy', env, verbose=1,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=1,
                    goal_selection_strategy="future",
                    online_sampling=False,
                    max_episode_length=1_000,
                ),
                tensorboard_log="logs/albert"
                )
    #model.learn(total_timesteps=1_000_000, log_interval=4, tb_log_name="SACHER_sparse_explo",
    #            callback=eval_callback)
    #model.save('models/albert/SACHER_sparse_explo')
    env.close()

    print("Starting evaluation")
    del env
    eval_env = gym.make("albert-reacher-vel-v0", dt=0.01, render=True)
    eval_env = Monitor(eval_env)
    n_episodes = 10
    n_steps = 5_000

    model = SAC.load('models/albert/SACHER_sparse.zip', env=eval_env)
    #model = SAC.load('models/debug/training/albert/best_model.zip', env=eval_env)

    durations = []
    for e in range(n_episodes):
        ob = eval_env.reset()

        for i in range(n_steps):
            #start = time.time()
            action, _states = model.predict(ob, deterministic=True)
            #end = time.time()
            #duration = end - start
            #durations.append(duration)
            ob, rew, done, infos = eval_env.step(action)
            if i % 100 == 1:
                print(i, ob['desired_goal'], ob['achieved_goal'], rew, done, infos)
            if done:
                print(i, rew, done, infos)
                break
    #durations = np.array(durations)
    pd.DataFrame(durations).to_csv('nn_durations.csv')


if __name__ == "__main__":
    main()
