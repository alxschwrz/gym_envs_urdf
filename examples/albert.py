import gym
import urdfenvs.albert_reacher
import numpy as np
import warnings


def main():
    env = gym.make("albert-reacher-vel-v0", dt=0.01, render=True)
    action = np.zeros(9)
    action[0] = 0.0
    action[1] = 0.0
    action[5] = -0.0
    action[7] = 0.1
    n_episodes = 15
    n_steps = 1_000
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    for ep in range(n_episodes):
        ob = env.reset()
        for i in range(n_steps):
            action = env.action_space.sample()
            ob, rew, _, _ = env.step(action)
            if i % 10 == 1:
                print(i, action[0:3], ob['achieved_goal'], ob['desired_goal'], rew)


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        main()
