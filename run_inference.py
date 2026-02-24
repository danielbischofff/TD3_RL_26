from __future__ import annotations

import os
import uuid
import numpy as np

import hockey.hockey_env as h_env
from comprl.client import Agent
from td3 import TD3_agent

# Video writing
try:
    import imageio.v2 as imageio
except Exception:
    import imageio


# -------------------------
# Your agents (guide-style)
# -------------------------

class RandomAgent(Agent):
    """A hockey agent that simply uses random actions (4D for HockeyEnv_BasicOpponent)."""
    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"


class HockeyAgent(Agent):
    """Wrapper around BasicOpponent; for HockeyEnv_BasicOpponent this is NOT needed
    (env already contains the opponent). Kept here for completeness."""
    def __init__(self, weak: bool) -> None:
        super().__init__()
        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # BasicOpponent.act expects obs for agent-two coordinate system normally,
        # so use with caution. For local eval vs weak opponent, prefer the env wrapper.
        return self.hockey_agent.act(observation).tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"

class SelfPlayOpponent:
    """Opponent that uses a TD3 policy (same interface as BasicOpponent.act)."""
    def __init__(self, ckpt_path: str):
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.num_actions
        act_bounds = (env.action_space.low[0], env.action_space.high[0])
        self.td3 = TD3_agent(obs_dim=obs_dim, act_dim=act_dim, act_bounds=act_bounds, ckpt_path=ckpt_path)

    def act(self, obs2):
        return self.td3.act(obs2)

class HockeyAgent_TD3(Agent):
    """Uses your TD3_agent wrapper to produce a 4D action for HockeyEnv_BasicOpponent."""
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        # IMPORTANT: for HockeyEnv_BasicOpponent, action is 4D (player1 only).
        # We'll still construct TD3_agent using the base env class for dims.
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.num_actions
        act_bounds = (env.action_space.low[0], env.action_space.high[0])
        self.td3 = TD3_agent(obs_dim=obs_dim, act_dim=act_dim,act_bounds=act_bounds, ckpt_path="checkpoints/td3_ckp_04_so.pt")

    def get_step(self, observation: list[float]) -> list[float]:
        return self.td3.act(observation).tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"


# -------------------------
# Local evaluation runner
# -------------------------

def run_local_games(
    agent: Agent,
    n_games: int = 20,
    video_path: str = "./results/hockey_vs_weak_all.mp4",
    fps: int = 50,
    score_factor: float = 0.25,
    seed: int = 0,
    opponnent: str = "weak",
    self_ckpt: str | None = None,   # <-- add
):
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- choose env/opponent ---
    if opponnent == "self":
        env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL, keep_mode=True)
        assert self_ckpt is not None, "Provide self_ckpt for self-play."
        player2 = SelfPlayOpponent(self_ckpt)
    else:
        env = h_env.HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=(opponnent != "strong"))
        player2 = None  # built-in

    wins = losses = ties = 0
    writer = imageio.get_writer(video_path, fps=fps)

    try:
        for ep in range(n_games):
            obs1, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

            # for self-play we need obs2 as well
            if opponnent == "self":
                obs2 = env.obs_agent_two()

            fake_game_id = uuid.uuid4().int.to_bytes(16, byteorder="big", signed=False)
            agent.on_start_game(fake_game_id)

            terminated = truncated = False
            ep_return = 0.0

            writer.append_data(env.render(mode="rgb_array"))

            while True:
                a1 = np.asarray(agent.get_step(obs1.tolist()), dtype=np.float32)

                if opponnent == "self":
                    a2 = np.asarray(player2.act(obs2), dtype=np.float32)
                    action = np.hstack([a1, a2])  # (8,)
                    obs1, reward, terminated, truncated, info = env.step(action)
                    obs2 = env.obs_agent_two()
                else:
                    obs1, reward, terminated, truncated, info = env.step(a1)

                ep_return += float(reward)
                writer.append_data(env.render(mode="rgb_array"))

                if terminated or truncated:
                    break

            winner = info.get("winner", 0)
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                ties += 1

            agent.on_end_game(winner == 1, [float(ep_return), 0.0])

    finally:
        writer.close()
        env.close()

    total = wins + losses + ties
    score = (wins - losses) * score_factor
    print("\n=== Summary ===")
    print(f"Games: {total}")
    print(f"Wins / Losses / Ties: {wins} / {losses} / {ties}")
    print(f"Score = (wins - losses) * factor = ({wins} - {losses}) * {score_factor} = {score:.3f}")
    print(f"Video saved to: {os.path.abspath(video_path)}")




if __name__ == "__main__":
    ckpt = "checkpoints/td3_ckp_04_so.pt"
    agent = HockeyAgent_TD3(ckpt_path=ckpt)

    run_local_games(
        agent=agent,
        n_games=10,
        video_path="./results/hockey_td3_vs_td3_all.mp4",
        fps=50,
        score_factor=0.25,
        seed=0,
        opponnent="self",
        self_ckpt=ckpt,  # opponent ckpt (can be different!)
    )
