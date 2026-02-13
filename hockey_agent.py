import hockey.hockey_env as h_env
from comprl.client import Agent

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