import gymnasium as gym
from ppo import PPO
from matplotlib import pyplot as plt

env = gym.make("CartPole-v1")
agent = PPO(env=env,
            RNN=False,
            total_timestep=1000,
            batch_size=10,
            max_timesteps_per_ep=100,
            epochs=100,
            gamma = 0.95,
            epsilon = 0.2,
            lr = 0.99
            )

agent.learn()