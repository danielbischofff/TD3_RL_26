import hockey.hockey_env as h_env
# from comprl.client import Agent
from td3 import TD3_agent
import numpy as np
import torch
import os
import re

# class HockeyAgent_TD3(Agent):
#     """Uses your TD3_agent wrapper to produce a 4D action for HockeyEnv_BasicOpponent."""
#     def __init__(self, ckpt_path: str) -> None:
#         super().__init__()
#         # IMPORTANT: for HockeyEnv_BasicOpponent, action is 4D (player1 only).
#         # We'll still construct TD3_agent using the base env class for dims.
#         env = h_env.HockeyEnv()
#         from td3 import TD3_agent
#         env = h_env.HockeyEnv()
#         obs_dim = env.observation_space.shape[0]
#         act_dim = env.num_actions
#         act_bounds = (env.action_space.low[0], env.action_space.high[0])
#         self.td3 = TD3_agent(obs_dim=obs_dim, act_dim=act_dim,act_bounds=act_bounds, ckpt_path="./checkpoints")

#     def get_step(self, observation: list[float]) -> list[float]:
#         return self.td3.act(observation).tolist()

#     def on_start_game(self, game_id) -> None:
#         game_id = uuid.UUID(int=int.from_bytes(game_id))

#     def on_end_game(self, result: bool, stats: list[float]) -> None:
#         text_result = "won" if result else "lost"


def list_td3_ckpts(ckpt_dir: str):
    CKPT_RE = re.compile(r"^td3_ckp_(\d+)(?:_.*)?\.pt$")
    """Return list of (step:int, path:str), sorted oldest->newest."""
    items = []
    for fn in os.listdir(ckpt_dir):
        m = CKPT_RE.match(fn)
        if m:
            step = int(m.group(1))
            items.append((step, os.path.join(ckpt_dir, fn)))
    items.sort(key=lambda x: x[0])
    return items

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

        ckpt = self.get_new_ckpt()
        if self.current_name == "td3_v1":
            self.opponents["td3_v1"].re_init_actor(ckpt)

        return self.current_name

    def act(self, observation):
        """Act using the currently selected opponent."""
        if self.current_agent is None:
            self.new_episode()

        return self.current_agent.act(observation)

    def get_new_ckpt(self):
        ckpts = list_td3_ckpts("./checkpoints")
        n = len(ckpts)

        offset_lo = int(np.floor(0.15 * n))
        offset_hi = int(np.floor(0.20 * n))
        offset_hi = max(offset_hi, offset_lo + 1)

        offset = self.rng.integers(offset_lo, offset_hi + 1)  # inclusive hi
        step, path = ckpts[-offset]

        return path
    
    def get_new_ckpt(self):
        ckpts = list_td3_ckpts("./checkpoints")
        n = len(ckpts)
        if n == 0:
            return self.opponents["td3_v1"].ckpt_path

        if n < 10:
            pool = ckpts[:-1] if n > 1 else ckpts
            step, path = pool[int(self.rng.integers(0, len(pool)))]
            return path

        offset_lo = max(1, int(np.floor(0.15 * n)))
        offset_hi = max(offset_lo + 1, int(np.floor(0.20 * n)))

        offset_hi = min(offset_hi, n)  # cap so -offset is valid
        offset = int(self.rng.integers(offset_lo, offset_hi + 1))  # inclusive hi

        step, path = ckpts[-offset]
        return path