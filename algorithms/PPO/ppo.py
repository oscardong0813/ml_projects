import gymnasium as gym
import penv as penv

from network import *

class PPO:
    def __init__(self, env, total_timestep=10, batch_size=2, gamma = 0.95):
        #extract env info
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.total_timestep = total_timestep
        self.batch_size = batch_size

        self.gamma = gamma #discount rate for rewards

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
        batch_states = []
        batch_acts = []
        batch_log_probs =[]
        batch_rews = []
        batch_disc_rew = []
        batch_lens = []

        t = 0
        # while t < self.batch_size:
        #     ep_rews = []
        #     state = self.env.reset()
        #     done = False
        #     break
        pass

    def choose_action(self,state):
        '''
        :param state:
        :return: action to take and log probability of the action

        only softmax used, no particular distribution used.
        '''
        state = torch.tensor([state], dtype=torch.float)
        dist = self.actor(state)
        action = dist.sample()
        # log_probs = torch.sequeeze(dist.log_prob(action)).item()
        # action = torch.sequeeze(action).item()
        log_prob = dist.log_prob(action)
        print('action sampled ', action, ' and log prob of the action ', log_prob)
        return action, log_prob

    def compute_disc_rew(self, batch_rews):
        '''
        :param batch_rews:an array of rewards from the batch
        :return: discounted rewards using gamma discount rate
        '''
        batch_disc_rew = []
        for ep_rew in reversed(batch_rews):
            disc_rew = 0
            for rew in reversed(ep_rew):
                disc_rew = rew + disc_rew * self.gamma
                batch_disc_rew.insert(0, disc_rew)

        batch_disc_rew = torch.tensor(batch_disc_rew, dtype=torch.float)
        return batch_disc_rew
    def compute_adv_est(self):
        pass

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    print(env.observation_space.shape, env.action_space.shape)
    env.reset() #initial state, [cart position, cart velocity, pole angle, pole angular velocity]
    next_state, rewards, terminated, truncated, info = env.step(0)
    env.render()

