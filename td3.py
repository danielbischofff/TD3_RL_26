import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class TD3_trainer():
    def __init__(self, obs_dim, act_dim, act_bounds, resume_ckpt = None, resume_buffer=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = act_dim
        self.observation_space_dim = obs_dim
        self.resume_ckpt = resume_ckpt
        self.resume_buffer = resume_buffer
        self.config = {
            "batch_size" : 100,
            "gamma" : 0.99,
            "tau" : 0.005,
            "policy_delay" : 2,
            "warm_up" : 10000,
            "max_episodes" : 10000,
            "exploration_noise_s" : 0.1,
            "target_smoothing_noise_s" : 0.2,
            "target_noise_clipping": 0.5,
            "lr_critic" : 1e-4,
            "lr_actor" : 1e-4,
            "checkpoint_path": "./checkpoints",
            "checkpoint_interval": 1000
        }
        self.start_timestep = 0
        self.max_episodes = self.config["max_episodes"]
        self.action_bounds = act_bounds
        self.buffer = ReplayBuffer(state_dim=self.observation_space_dim, action_dim=self.action_dim)
        self.initiate_models()

    def initiate_models(self):
        self.actor = Actor(self.observation_space_dim, self.action_dim, max_action=1).to(self.device)
        self.critic_1 = Critic(self.observation_space_dim, self.action_dim).to(self.device)
        self.critic_2 = Critic(self.observation_space_dim, self.action_dim).to(self.device)

        self.actor_target = Actor(self.observation_space_dim, self.action_dim, max_action=1).to(self.device)
        self.critic_1_target = Critic(self.observation_space_dim, self.action_dim).to(self.device)
        self.critic_2_target = Critic(self.observation_space_dim, self.action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.config["lr_critic"])
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.config["lr_critic"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config["lr_actor"])

        if self.resume_ckpt:
            self.resume_from_check(self.resume_ckpt)
   

    def select_random_action(self):
        return np.random.uniform(self.action_bounds[0],self.action_bounds[1], self.action_dim).tolist()

    def select_action_with_policy(self, state,  add_noise=True):

        # check if state is numpy and transform to tensor
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_t = self.actor(state_t)
        action = action_t.cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, self.config["exploration_noise_s"], size=self.action_dim)
            action = action + noise

        # clip to env bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1]) # FIXME maybe we need this

        return action.astype(np.float32)
    
    def critic_update(self, state, action, reward, next_state, done):
        
        # ---- Compute targets (no gradients) ----
        with torch.no_grad():
            # generates noise
            noise = torch.randn_like(action) * self.config["target_smoothing_noise_s"]
            noise = noise.clamp(
                -self.config["target_noise_clipping"],
                self.config["target_noise_clipping"]
            )

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(self.action_bounds[0], self.action_bounds[1])

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)

            target_Q = reward + self.config["gamma"]* (1 - done) * torch.min(target_Q1, target_Q2)

        # ---- Critic prediction values ----
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # ---- Critic loss ----
        loss = torch.nn.SmoothL1Loss()
        critic_1_loss = loss(current_Q1, target_Q)
        critic_2_loss = loss(current_Q2, target_Q)
        

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
        }

    def actor_target_update(self, state_batch):

        # Actor loss: maximize Q_1(s, π(s))  -> minimize -Q_1(...)
        actor_loss = - self.critic_1(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft update of all target networks -----
        with torch.no_grad():
            # actor target
            tau = self.config["tau"]
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # critic_1 target
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # critic_2 target (if you have a second critic, as in TD3)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return actor_loss

    def save_checkpoint(self, step, name="", small=False):
        path = self.config["checkpoint_path"]
        if small:
            checkpoint = {
                "actor": self.actor.state_dict(),
                "config": self.config,
            }
        else:
            checkpoint = {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer_1": self.critic_optimizer_1.state_dict(),
                "critic_optimizer_2": self.critic_optimizer_2.state_dict(),
                "replay_buffer": self.buffer,
                "timestep": step,
                "config": self.config,
            }
        torch.save(checkpoint, f"{path}/td3_ckp_{step}_{name}.pt")
    
    def resume_from_check(self, path):
    
        ckpt = torch.load(path, map_location=self.device, weights_only = False)
        self.config = ckpt["config"]

        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
        self.critic_2_target.load_state_dict(ckpt["critic_2_target"])

        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer_1.load_state_dict(ckpt["critic_optimizer_1"])
        self.critic_optimizer_2.load_state_dict(ckpt["critic_optimizer_2"])


        if self.resume_buffer:
            self.buffer = ckpt["replay_buffer"]
            self.start_timestep = int(ckpt["timestep"])


class TD3_agent():
    def __init__(self, obs_dim, act_dim, act_bounds, ckpt_path):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = act_dim
        self.observation_space_dim = obs_dim
        self.action_bounds = act_bounds

        self.actor = Actor(self.observation_space_dim, self.action_dim).to(self.device)
        self.init_actor()
        self.actor.eval()

    def act(self, observation):

        # check if state is numpy and transform to tensor
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        observation_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_t = self.actor(observation_t)
        action = action_t.cpu().numpy()[0]

        # clip to env bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return action.astype(np.float32)
    
    def init_actor(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])

    def re_init_actor(self, ckpt):
        ckpt = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])



class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, max_action=1):

        super().__init__()
        self.l1 = nn.Linear(observation_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            return self.max_action * torch.tanh(self.l3(x))
    

class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(observation_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)    
        
    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        return self.l3(x)
    
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=bool)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

class ReplayPriorityBuffer:
    def __init__(self, state_dim, action_dim, alpha, beta_start, beta_frames, eps ,max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=bool)

        # PER
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1

        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):

        prios = self.priorities[:self.size] + self.eps
        probs = prios ** self.alpha
        probs /= probs.sum()

        idx = np.random.randint(0, self.size, size=batch_size, p=probs)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )