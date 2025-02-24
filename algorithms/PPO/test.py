import gymnasium as gym
from ppo import PPO
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0", goal_velocity=0.1)
agent = PPO(env=env,
            RNN=False,
            total_timestep=100000,
            batch_size=100,
            max_timesteps_per_ep=1600,
            epochs=10,
            gamma = 0.95,
            epsilon = 0.2,
            lr = 0.99
            )

agent.learn()
print(agent.logger['batch_rewards'])