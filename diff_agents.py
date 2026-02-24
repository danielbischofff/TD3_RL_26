import torch
import torch.nn as nn
import numpy as np

class SACAgent(object):
    """
    Agent implementing SAC algorithm with NN function approximation.
    """
    def __init__(self, obs_dim, act_dim, ckpt_path, act_bounds):

        self._obs_dim= obs_dim
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._action_n = 8
        self._config = {
            "discount": 0.99,
            "initial_alpha": 0.2, # Alpha: relative importance of the entropy term against the reward 
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": 0.0001, #oder eine Null mehr?
            "learning_rate_V": 0.0001,
            "learning_rate_Q": 0.0001,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "update_target_every": 1000,
        }

        self.act_bounds = act_bounds
 
        # Policy network
        self.policy = policyFunction(observation_dim=self._obs_dim,  
                                     action_dim=self._action_n,                     
                                     hidden_sizes= self._config["hidden_sizes_actor"],
                                     learning_rate=self._config["learning_rate_actor"]).to(self.device)
        
        self.init_actor()

    def act(self, observation): # arguments schould be torch tensors

        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        observation_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_prerescale, _ = self.policy.sample(observation_t)  
        action = action_prerescale.cpu().numpy()
        action = self.act_bounds[0] + (action + 1.0) / 2.0 * (self.act_bounds[1] - self.act_bounds[0]) # Rescale from [-1, 1] to env action space
        return action[0]


    def state(self):
        return  self.policy.state_dict()
    
    def init_actor(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt[3])



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