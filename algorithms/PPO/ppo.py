import gymnasium as gym
from gymnasium.spaces import Discrete
import penv as penv
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from network import *

class PPO:
    def __init__(self, env, RNN = False, total_timestep=15, batch_size=5, max_timesteps_per_ep = 5, epochs = 5, gamma = 0.95, epsilon = 0.2, lr = 0.99):
        #extract env info
        self.env = env
        self.state_dim = env.observation_space.shape[0]

        #if use RNN networks
        self.RNN = RNN
        #checking if action space is discrete
        if isinstance(env.action_space, Discrete):
            self.act_dim = env.action_space.n
            self.discrete_agent = True
        else:
            self.act_dim = env.action_space.shape[0]
            self.discrete_agent = False
        self.total_timestep = total_timestep
        self.batch_size = batch_size
        self.max_timsteps_per_episode = max_timesteps_per_ep
        self.epochs = epochs

        self.lr = lr
        self.gamma = gamma #discount rate for rewards
        self.epsilon = epsilon #clip in objective loss function.

        #STEP 1 (initial policy parameters and initial value function parameters) initialize actor and critic networks
        self.actor = ActorNet(self.state_dim,self.act_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic = CriticNet(self.state_dim)
        self.critic_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.logger = {
            'batch_lens':[],
            'batch_rewards':[],
            'actor_losses':[],
            'batch_acts':[]
        }

    def learn(self):
        #STEP 2 (for k = 0, 1, 2, ... do) train for a total of n timesteps.
        timestep_sofar = 0
        while timestep_sofar < self.total_timestep:
            #STEP 3 collect set of trajectories by running the current policy.
            #use roll_out method to collect STEP 4 rewards-to-go,
            batch_states, batch_acts, batch_log_probs, batch_disc_rews, batch_lens = self.rollout()

            # STEP 5advantage estimates
            adv_est = self.compute_adv_est(batch_states, batch_disc_rews)

            for _ in range(self.epochs):
                # STEP 6 update the policy by maximizing the ppo-clip objective (via stochastic gradient ascent with Adam)
                curr_log_probs = self.compute_curr_log_probs(batch_states,batch_acts)
                ratio_policies = torch.exp(curr_log_probs - batch_log_probs) #logA/logB = logA - logB

                unclipped = ratio_policies * adv_est #ratio*Adv_est
                clipped = torch.clamp(ratio_policies, 1-self.epsilon, 1+self.epsilon) * adv_est #clipped (1-ep<=ratio<=1+ep)*Adv_est
                actor_loss = (-torch.min(unclipped, clipped)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Step 7 fit value function by regression on mean-sequared error
                v = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(v, batch_disc_rews)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

            timestep_sofar += np.sum(batch_lens)

    def rollout(self):
        # print('starting rollout')
        batch_states = []
        batch_acts = []
        batch_log_probs =[]
        batch_rews = []
        # batch_disc_rews = []
        batch_lens = []

        t = 0
        while t < self.batch_size:
            ep_rews = []
            state = self.env.reset()
            state = state[0]
            for ep_timestep in range(self.max_timsteps_per_episode):
                t += 1
                # print('adding to batch states ', state)
                # batch_states.append(torch.tensor(state,dtype=torch.float))
                batch_states.append(state)
                sampled_action, log_prob = self.choose_action(state)
                # print('sampled action ', sampled_action.item(), ' log_prob ', log_prob)
                state, rew, terminated, truncated, info = self.env.step(sampled_action.item())
                # print('stepped ', t)
                ep_rews.append(rew)
                batch_acts.append(sampled_action)
                batch_log_probs.append(log_prob)

                if terminated or truncated:
                    break

            batch_lens.append(ep_timestep +1)
            batch_rews.append(ep_rews)

        # print('batched states ', batch_states)
        batch_states = torch.tensor(batch_states, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_disc_rews = self.compute_disc_rew(batch_rews)
        # print('batch states ', batch_states)
        # print('batch rew ', batch_rews, batch_disc_rews)

        self.logger['batch_rewards'].append(batch_rews)
        self.logger['batch_lens'].append(batch_lens)
        self.logger['batch_acts'].append(batch_acts)

        return batch_states, batch_acts, batch_log_probs,batch_disc_rews, batch_lens

    def evaluate(self, batch_states, batch_acts):
        v = self.critic(batch_states).squeeze()

        dist = Categorical(self.actor(batch_states))
        log_probs = dist.log_prob(batch_acts)
        return v, log_probs

    def choose_action(self,state):
        '''
        :param state:
        :return: action to take and log probability of the action

        only softmax used, no particular distribution used.
        '''
        # print('input state ', type(state), state)
        state = torch.tensor(state, dtype=torch.float)

        # print(self.actor(state))
        dist = Categorical(self.actor(state))
        action = dist.sample()
        # log_probs = torch.sequeeze(dist.log_prob(action)).item()
        # action = torch.sequeeze(action).item()
        log_prob = dist.log_prob(action)
        # print('action sampled ', action, ' and log prob of the action ', log_prob)
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

    def compute_adv_est(self, batch_states, batch_disc_rews):
        # using detach() so that critic vals are determined before epoch updates network
        critic_val_for_adv_est = self.critic(batch_states).squeeze().detach()
        adv_est = batch_disc_rews - critic_val_for_adv_est

        normalized = (adv_est - adv_est.mean()) / (adv_est.std() + 1e-10)

        return normalized

    def compute_curr_log_probs(self, batch_states, batch_actions):
        dist = Categorical(self.actor(batch_states))
        log_probs = dist.log_prob(batch_actions)

        return log_probs


if __name__ == '__main__':
    env1 = gym.make('CartPole-v1', render_mode='rgb_array')
    agent1 = PPO(env1)
    agent1.learn()
    print(agent1.logger)

    env2 = gym.make('MountainCar-v0')
    agent2 = PPO(env2)
    agent2.learn()
    print(agent2.logger)



