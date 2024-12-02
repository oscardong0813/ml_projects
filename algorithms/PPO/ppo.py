import gymnasium as gym
import penv as penv

from network import ActorNet, CriticNet

class PPO:
    def __init__(self, env, total_timestep=10000, batch_size=200):
        #extract env info
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.total_timestep = total_timestep
        self.batch_size = batch_size

        #STEP 1 (initial policy parameters and initial value function parameters) initialize actor and critic networks
        self.actor = ActorNet(self.state_dim,self.act_dim)
        self.critic = CriticNet(self.state_dim)

    def learn(self):
        #STEP 2 (for k = 0, 1, 2, ... do) train for a total of n timesteps.
        timestep_sofar = 0
        while timestep_sofar < self.total_timestep:
            #STEP 3 collect set of trajectories by running the current policy.
            #use roll_out method to collect STEP 4 rewards-to-go, STEP 5advantage estimates
            #STEP 6 update the policy by maximizing the ppo-clip objective (via stochastic gradient ascent with Adam)
            #Step 7 fit value function by regression on mean-sequared error
            break

    def rollout(self):
        pass
    def rewardsTGo(self):
        pass
    def advEst(self):
        pass

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    print(env.observation_space.shape, env.action_space.shape)
    env.reset() #initial state, [cart position, cart velocity, pole angle, pole angular velocity]
    next_state, rewards, terminated, truncated, info = env.step(0)
    env.render()

