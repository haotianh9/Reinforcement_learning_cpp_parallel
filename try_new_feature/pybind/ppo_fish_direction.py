# ------------------------------------------------------------------
# required package: openAI gym, PyTorch
# ------------------------------------------------------------------
import torch
import ppo_fish
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import os
import fish_evasion as fish


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    # structure to store data for each update
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # operations per decision time step
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = torch.squeeze(self.actor(state))

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def main(k):
    path = './direction_BS_woNorm/150/{}'.format(k)
    if not os.path.exists(path):
        os.makedirs(path)
    ############## Hyperparameters ##############
    env_name = "fishEvasion-v0"  # used when creating the en
    render = False  # render the environment in training if true
    # solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 27  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 150  # max timesteps in one episode

    update_timestep = 4050  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    env = fish.FishEvasionEnv(dt=0.1)

    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)

    # get observation and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # memory = Memory()
    memory = ppo_fish.MemoryNN()
    ppo = ppo_fish.PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    # ------------------------------------------------------------------
    # start training from an existing policy
    # ppo.policy_old.load_state_dict(torch.load('./direction_policy/PPO_{}_{:06d}.pth'.format(env_name,4380),map_location=device))
    # ppo.policy.load_state_dict(torch.load('./direction_policy/PPO_{}_{:06d}.pth'.format(env_name,4380),map_location=device))
    # ------------------------------------------------------------------

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        # ------------------------------------------------------------------
        # set a specific distribution for beta
        # beta0 = angle_normalize(i_episode*3,center = 0)
        # print(beta0)
        # ------------------------------------------------------------------
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(torch.tensor(state).float(), memory)
            # action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Storing reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if it is time
            # ------------------------------------------------------------------
            if time_step % update_timestep == 0:
                ppo.update(memory)
                # memory.clear_memory()
                memory.clear()
                time_step = 0
            # ------------------------------------------------------------------
            running_reward += reward
            if render:
                env.render()
            # break if episode ends
            if done:
                break
        avg_length += t

        # ------------------------------------------------------------------
        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_forwardWoPos_solved_{}.pth'.format(env_name))
        #     break
        # ------------------------------------------------------------------

        # save every 50 episodes
        if i_episode % 50 == 0:
            torch.save(ppo.policy.state_dict(), path + '/PPO_{}_direction{:06d}.pth'.format(env_name, i_episode))

            # ------------------------------------------------------------------
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
        # ------------------------------------------------------------------


if __name__ == '__main__':
    print('direction150')
    # training for 25 times
    for k in range(1, 25):
        main(k)


# wrap angles about a given center
def angle_normalize(x, center=0):
    return (((x + np.pi - center) % (2 * np.pi)) - np.pi + center)
