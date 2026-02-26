######################### SAC #######################################

# The architecture is inspired by the solution for the DDPG algorithm of Ex-Sheet 9. 

import torch
import torch.nn as nn
import random
import numpy as np
import gymnasium as gym
import optparse
import pickle
import hockey.hockey_env as h_env
from td3 import TD3_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# buffer
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=100000):
        self.max_size = max_size
        self.pointer = 0
        self.size = 0
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs_new_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.bool_)
         
    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.pointer] = obs
        self.acts_buf[self.pointer] = act
        self.rews_buf[self.pointer] = rew
        self.obs_new_buf[self.pointer] = next_obs
        self.done_buf[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        if batch_size > self.size:
            batch_size = self.size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs_buf[idxs],
                self.acts_buf[idxs],
                self.rews_buf[idxs],
                self.obs_new_buf[idxs],
                self.done_buf[idxs])
    
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()

class VFunction(Feedforward):
    def __init__(self, observation_dim, hidden_sizes=[256,256],
                 learning_rate = 0.0001):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.MSELoss() # L2 loss

    def fit(self, observations, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        pred = self.forward(observations)
        # Compute Loss
        loss = self.loss(pred, targets)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class QFunction(Feedforward): 
    def __init__(self, observation_dim, action_dim, hidden_sizes=[256,256],
                 learning_rate = 0.0001):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.MSELoss() #L2 loss

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        pred = self.Q_value(observations, actions)
        # Compute Loss
        loss = self.loss(pred, targets)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))


class policyFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[1], action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # for stability 
        std = log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        noise = torch.randn_like(mean)
        pre_tanh_action = mean + std * noise
        action = torch.tanh(pre_tanh_action)  # Squashing for bounded actions
        distr = torch.distributions.Normal(mean, std)
        #log_prob = distr.log_prob(action).sum(dim=-1)
        log_prob = distr.log_prob(pre_tanh_action) - torch.log(1 - action.pow(2) + 1e-6) # log-prob-correction for tanh squashing
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class SACAgent(object):
    """
    Agent implementing SAC algorithm with NN function approximation.
    """
    def __init__(self, observation_space, action_space):

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "discount": 0.99,
            "initial_alpha": 0.2, # Alpha: relative importance of the entropy term against the reward 
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": 0.0001, 
            "learning_rate_V": 0.0001,
            "learning_rate_Q": 0.0001,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "update_target_every": 1000,
        }

        self.buffer = ReplayBuffer(self._obs_dim, self._action_n, max_size=self._config["buffer_size"])
        self.entropy = None # for logging

        # learnable alpha parameter
        self.alpha = self._config["initial_alpha"]
        self.target_entropy = -self._action_n  # heuristic value from the SAC paper
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha), dtype=torch.float32))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.00001)

        # V network
        self.V = VFunction(observation_dim=self._obs_dim,
                           hidden_sizes=self._config["hidden_sizes_critic"],
                           learning_rate=self._config["learning_rate_V"]).to(device)
        
        # target V Network
        self.V_target = VFunction(observation_dim=self._obs_dim, 
                                 hidden_sizes=self._config["hidden_sizes_critic"],
                                 learning_rate=0).to(device)

        # Q Networks
        self.Q1 = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_Q"]).to(device)

        self.Q2 = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_Q"]).to(device)
        # Policy network
        self.policy = policyFunction(observation_dim=self._obs_dim,  
                                     action_dim=self._action_n,                     
                                     hidden_sizes= self._config["hidden_sizes_actor"],
                                     learning_rate=self._config["learning_rate_actor"]).to(device)

        self._copy_nets()
        self.train_iter = 0

    def _copy_nets(self):
        self.V_target.load_state_dict(self.V.state_dict())

    def act(self, observation): # arguments schould be torch tensors
        observation = torch.from_numpy(observation.astype(np.float32)).to(device)
        with torch.no_grad():
            action_prerescale, _ = self.policy.sample(observation)  
        action = action_prerescale.cpu().numpy()
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low) # Rescale from [-1, 1] to env action space
        return action

    def store_transition(self, obs, act, rew, obs_new, done):
        self.buffer.add(obs=obs, act=act, rew=rew, next_obs=obs_new, done=done)

    def state(self):
        return (self.V.state_dict(), self.Q1.state_dict(), self.Q2.state_dict(), self.policy.state_dict(), self.log_alpha.data)
    
    def optimizer_state(self):
        return (self.V.optimizer.state_dict(), self.Q1.optimizer.state_dict(), self.Q2.optimizer.state_dict(), self.policy.optimizer.state_dict(), self.alpha_optimizer.state_dict())

    def restore_state(self, state):
        self.V.load_state_dict(state[0])
        self.Q1.load_state_dict(state[1])
        self.Q2.load_state_dict(state[2])
        self.policy.load_state_dict(state[3])
        self.log_alpha.data.copy_(state[4])
        self.alpha = self.log_alpha.exp().item()
        self._copy_nets()

    def restore_optimizer_state(self, optimizer_state):
        self.V.optimizer.load_state_dict(optimizer_state[0])
        self.Q1.optimizer.load_state_dict(optimizer_state[1])
        self.Q2.optimizer.load_state_dict(optimizer_state[2])
        self.policy.optimizer.load_state_dict(optimizer_state[3])
        self.alpha_optimizer.load_state_dict(optimizer_state[4])

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter+=1
        if self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            s, a, rew, s_prime, done = self.buffer.sample(batch_size=self._config['batch_size'])
            s = to_torch(s).to(device) # s_t
            a = to_torch(a).to(device) # a_t
            rew = to_torch(rew).to(device) # reward  (batchsize,1)
            s_prime = to_torch(s_prime).to(device) # s_t+1
            done = to_torch(done).to(device) # done signal  (batchsize,1)

            # target for V function
            actions, log_probs = self.policy.sample(s)
            q1 = self.Q1.Q_value(s, actions)
            q2 = self.Q2.Q_value(s, actions)
            q_min = torch.min(q1, q2)
            v_targets = q_min - self.alpha * log_probs

            # optimize the V objective
            v_loss = self.V.fit(s, v_targets.detach())

            # target for Q functions
            gamma = self._config['discount']
            v_target_prime = self.V_target.forward(s_prime).detach()
            q_targets = rew + gamma * (1.0-done) * v_target_prime

            # optimize the Q1 and Q2 objective
            q1_loss = self.Q1.fit(s, a, q_targets)
            q2_loss = self.Q2.fit(s, a, q_targets)

            # optimize actor (policy) objective
            self.policy.optimizer.zero_grad()
            actions, log_probs = self.policy.sample(s)
            q1_pi = self.Q1.Q_value(s, actions)
            q2_pi = self.Q2.Q_value(s, actions)
            q_pi_min = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_probs - q_pi_min).mean()
            actor_loss.backward()
            self.policy.optimizer.step()
            self.entropy = -log_probs.mean().item() # for logging

            # optimize alpha (entropy temperature) objective
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            losses.append((v_loss, q1_loss, q2_loss, actor_loss.item(), alpha_loss.item()))

        return losses


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="HockeyEnv",
                         help='Environment (default %default)')
    optParser.add_option('-m', '--numepisodes',action='store', type='int',
                         dest='num_episodes',default=10000,
                         help='number of episodes (default %default)')
    optParser.add_option('-o', '--opponent',action='store', type='string',
                         dest='opponent',default="BasicOpponent_weak",
                         help='Opponent (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "HockeyEnv":
        env = h_env.HockeyEnv()
    else:
        raise ValueError("Unknown environment name")
    num_episodes = opts.num_episodes # max training episodes
    max_timesteps = 600         # max timesteps in one episode
    #############################################

    def save_buffer(buffer, filename):
        np.savez(filename,
                obs_buf=buffer.obs_buf,
                acts_buf=buffer.acts_buf,
                rews_buf=buffer.rews_buf,
                next_obs_buf=buffer.obs_new_buf,
                done_buf=buffer.done_buf,
                pointer=buffer.pointer,
                size=buffer.size)
    
    def load_buffer(buffer, filename):
        data = np.load(filename)
        buffer.obs_buf = data['obs_buf']
        buffer.acts_buf = data['acts_buf']
        buffer.rews_buf = data['rews_buf']
        buffer.obs_new_buf = data['next_obs_buf']
        buffer.done_buf = data['done_buf']
        buffer.pointer = int(data['pointer'])
        buffer.size = int(data['size'])

    def save_statistics(count): #FIXME
        with open(f"./results/SAC_fixed4_Report-{opts.opponent}-stat_{count}_3.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "losses": losses, "alphas": alphas, "entropies": entropies, "wins": wins}, f)

    sac = SACAgent(env.observation_space, env.action_space)
    
    # load saved model (FIXME: adjust paths)
    sac.restore_state(torch.load(f'SAC_fixed4_HockeyEnv-mixed_state_62.pth', map_location=device))
    #sac.restore_optimizer_state(torch.load(f'./results/SAC_fixed4_HockeyEnv-mixed_optimizer_62.pth', map_location=device))
    #load_buffer(sac.buffer, './results/SAC_fixed4_HockeyEnv-mixed_buffer_62.npz')
    
    def select_opponent():
        r = random.random()
        if r < 0.5:
            return "self-play"
        elif r < 0.8:
            return "TD3"
        else:
            return "BasicOpponent_strong"

    # initialize opponent
    def set_opponent(opponent_type):
        if opponent_type == "BasicOpponent_weak":
            return h_env.BasicOpponent()
        elif opponent_type == "BasicOpponent_strong":
            return h_env.BasicOpponent(weak=False)
        elif opponent_type == "self-play":
            player2 = SACAgent(env.observation_space, env.action_space)
            player2.restore_state(sac.state())
            return player2
        elif opponent_type == "TD3":
            act_bounds = (env.action_space.low[0], env.action_space.high[0])
            player2 = TD3_agent(obs_dim=env.observation_space.shape[0], act_dim=env.num_actions, act_bounds=act_bounds, ckpt_path="td3_ckp_mixed_05.pt") #TODO: adjust path
            return player2
        else:            
            raise ValueError("Unknown opponent type")

    if opts.opponent == "BasicOpponent_weak":
        player2 = h_env.BasicOpponent()
    elif opts.opponent == "BasicOpponent_strong":
        player2 = h_env.BasicOpponent(weak=False)
    elif opts.opponent == "self-play":
        player2 = SACAgent(env.observation_space, env.action_space)
        player2.restore_state(sac.state())
    elif opts.opponent == "TD3":
        act_bounds = (env.action_space.low[0], env.action_space.high[0])
        player2 = TD3_agent(obs_dim=env.observation_space.shape[0], act_dim=env.num_actions, act_bounds=act_bounds, ckpt_path="td3_ckp_mixed_05.pt") #TODO: adjust path
    elif opts.opponent == "mixed":
        player2 = set_opponent(select_opponent())
    else:            
        raise ValueError("Unknown opponent type")


    # logging variables
    rewards = []
    lengths = []
    losses = []
    alphas = []
    entropies = []
    wins = []
    timestep = 0
    save_count = 0 #FIXME:set number to the number of the last saved model

    # training loop
    for game in range(1, num_episodes+1):
        if game % 100 == 0 and opts.opponent == "mixed":
            player2 = set_opponent(select_opponent())
        obs, info = env.reset()
        #_ = env.render()
        obs_agent2 = env.obs_agent_two()

        total_reward=0
        episode_losses = []
        for t in range(max_timesteps):
            timestep += 1 
            done = False 
            a1 = sac.act(obs)
            a2 = player2.act(obs_agent2) 
            (obs_new, reward, done, trunc, info) = env.step(np.hstack([a1[:4],a2]))
            reward = reward #+ info["reward_puck_direction"] + info["reward_touch_puck"] # add rewards for intermediate goals
            total_reward+= reward
            sac.store_transition(obs, a1, reward, obs_new, done)
            obs=obs_new
            obs_agent2 = env.obs_agent_two()
            if 2*sac.buffer.size >= sac._config['batch_size']:
                loss = sac.train(1)
                episode_losses.extend(loss)
            if done or trunc: 
                wins.append(info['winner']) # =0 if draw, =1 if agent1 wins, =-1 if agent2 (opponent) wins
                break

        rewards.append(total_reward)
        lengths.append(t)
        alphas.append(sac.alpha)
        entropies.append(sac.entropy)
        if len(episode_losses) > 0:
            losses.append(np.mean(episode_losses, axis=0))
        else:
            losses.append(np.nan) # mean loss over all training steps in the episode

        # for self-play: update opponent policy
        if opts.opponent == "self-play" and game % 100 == 0:
             player2.restore_state(sac.state())


        # save every 2000 episodes
        if game % 1000 == 0: #FIXME
            save_count += 1
            torch.save(sac.state(), f'./results/SAC_fixed4_Report-{opts.opponent}_state_{save_count}_3.pth') #FIXME
            torch.save(sac.optimizer_state(), f'./results/SAC_fixed4_Report-{opts.opponent}_optimizer_{save_count}_3.pth')
            save_statistics(save_count)
            save_buffer(sac.buffer, f'./results/SAC_fixed4_Report-{opts.opponent}_buffer_{save_count}_3.npz')

    env.close()

if __name__ == '__main__':
    main()
