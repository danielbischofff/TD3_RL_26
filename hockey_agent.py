import hockey.hockey_env as h_env
from comprl.client import Agent
from td3 import TD3_agent
import numpy as np
import torch

class HockeyAgent_TD3(Agent):
    """Uses your TD3_agent wrapper to produce a 4D action for HockeyEnv_BasicOpponent."""
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        # IMPORTANT: for HockeyEnv_BasicOpponent, action is 4D (player1 only).
        # We'll still construct TD3_agent using the base env class for dims.
        env = h_env.HockeyEnv()
        from td3 import TD3_agent
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.num_actions
        act_bounds = (env.action_space.low[0], env.action_space.high[0])
        self.td3 = TD3_agent(obs_dim=obs_dim, act_dim=act_dim,act_bounds=act_bounds, ckpt_path="./checkpoints")

    def get_step(self, observation: list[float]) -> list[float]:
        return self.td3.act(observation).tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"


class RandomAgent():
    """A hockey agent that simply uses random actions."""
    def __init__(self, act_dim, act_bounds):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = act_dim
        self.action_bounds = act_bounds

    def act(self, observation: list[float]) -> list[float]:
        return np.random.uniform(self.action_bounds[0], self.action_bounds[1], self.action_dim).tolist()


class MixedAgent:
    def __init__(self, opponents: dict, probs: dict, seed: int = 0):

        self.opponents = opponents
        self.rng = np.random.default_rng(seed)

        self.names = list(probs.keys())
        p = np.array([probs[n] for n in self.names], dtype=np.float64)
        self.p = p / p.sum()

        self.current_name = None
        self.current_agent = None

    def new_episode(self):
        self.current_name = self.rng.choice(self.names, p=self.p)
        self.current_agent = self.opponents[self.current_name]
        return self.current_name

    def act(self, observation):
        """Act using the currently selected opponent."""
        if self.current_agent is None:
            self.new_episode()
        return self.current_agent.act(observation)