import gym
import urdfenvs.panda_reacher
import numpy as np
from stable_baselines3.sac import SAC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def main():
    gripper = False
    env = gym.make("panda-reacher-vel-v0", dt=0.01, render=False, gripper=gripper)
    eval_env = gym.make("panda-reacher-vel-v0", dt=0.01, render=False, gripper=gripper)
    check_env(env)
    env = Monitor(env)
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(env, eval_freq=2_500, n_eval_episodes=5,
                                 #best_model_save_path='models/debug/training/',
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
                tensorboard_log="logs/debug")
    #model.learn(total_timesteps=7_000, log_interval=4, #tb_log_name="SACHER_sparse",
    #            callback=eval_callback)
    #model.save('models/debug/SACHER_sparse')
    env.close()

    print("Starting evaluation")
    del env
    eval_env = gym.make("panda-reacher-vel-v0", dt=0.01, render=True, gripper=gripper, task_list=['full'])
    eval_env = Monitor(eval_env)
    gain = 1.1
    n_episodes = 10
    n_steps = 1_000

    #model = SAC.load('models/debug/SACHER_sparse.zip', env=eval_env)
    #import time
    #durations = []
    for e in range(n_episodes):
        ob = eval_env.reset()
        for i in range(n_steps):
            if i == 500:
                gain = 0.1
            action = eval_env.action_space.sample()
            action[:] = 0
            action[2] = gain * 0.1
            action[3] = gain * -0.08
            action[5] = 0.0
            #action[-2] = 1.0
            #start = time.time()
            #action, _states = model.predict(ob, deterministic=True)

            #end = time.time()
            #duration = end-start
            #durations.append(duration)
            #print(action)
            ob, rew, done, infos = eval_env.step(action)
            if i % 100 == 1:
                print(i, ob['desired_goal'], ob['achieved_goal'], rew, done, infos)
            if done:
                print(i, rew, done, infos)
                break
                #print(i, ob['achieved_goal'], ob['desired_goal'], rew)
    #durations = np.array(durations)
    #import pandas as pd
    #pd.DataFrame(durations).to_csv('NN_durations.csv')

if __name__ == "__main__":
    main()
