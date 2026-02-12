from __future__ import annotations

import os
import uuid
import numpy as np

import hockey.hockey_env as h_env
from comprl.client import Agent

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


class HockeyAgent_TD3(Agent):
    """Uses your TD3_agent wrapper to produce a 4D action for HockeyEnv_BasicOpponent."""
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        # IMPORTANT: for HockeyEnv_BasicOpponent, action is 4D (player1 only).
        # We'll still construct TD3_agent using the base env class for dims.
        env = h_env.HockeyEnv()
        from td3 import TD3_agent
        self.td3 = TD3_agent(env=env, ckpt_path=ckpt_path)

    def get_step(self, observation: list[float]) -> list[float]:
        return self.td3.act(observation).tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"


# -------------------------
# Local evaluation runner
# -------------------------

def run_local_games_vs_weak(
    agent: Agent,
    n_games: int = 20,
    video_path: str = "./results/hockey_vs_weak.mp4",
    fps: int = 50,
    score_factor: float = 0.25,
    seed: int = 0,
    opponnent: str = "weak",
):
    """
    Runs games locally vs BasicOpponent using HockeyEnv_BasicOpponent.
    Records ALL episodes into ONE MP4 (concatenated) and prints win/loss stats + score.
    """
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    weak_opponent = (opponnent.lower() == "weak")
    env = h_env.HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=weak_opponent)

    wins = losses = ties = 0

    # Stream frames directly to disk (no huge RAM usage)
    writer = imageio.get_writer(video_path, fps=fps)

    try:
        # optional reproducibility
        rng = np.random.default_rng(seed)

        for ep in range(n_games):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

            fake_game_id = uuid.uuid4().int.to_bytes(16, byteorder="big", signed=False)
            agent.on_start_game(fake_game_id)

            terminated = truncated = False
            ep_return = 0.0

            # record initial frame
            writer.append_data(env.render(mode="rgb_array"))

            while True:
                action = agent.get_step(obs.tolist())
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))

                ep_return += float(reward)

                # record every step
                writer.append_data(env.render(mode="rgb_array"))

                if terminated or truncated:
                    break

            winner = info.get("winner", 0)
            if winner == 1:
                wins += 1
                result_bool = True
            elif winner == -1:
                losses += 1
                result_bool = False
            else:
                ties += 1
                result_bool = False

            agent.on_end_game(result_bool, [float(ep_return), 0.0])

    finally:
        writer.close()
        env.close()

    total = wins + losses + ties
    score = (wins - losses) * score_factor

    print(f"\nSaved video to: {video_path}")
    print("\n=== Summary ===")
    print(f"Games: {total}")
    print(f"Wins / Losses / Ties: {wins} / {losses} / {ties}")
    print(f"Score = (wins - losses) * factor = ({wins} - {losses}) * {score_factor} = {score:.3f}")


if __name__ == "__main__":
    # Choose one:
    # agent = RandomAgent()
    agent = HockeyAgent_TD3(ckpt_path="./checkpoints")

    run_local_games_vs_weak(
        agent=agent,
        n_games=20,
        video_path="hockey_vs_weak.mp4",
        fps=50,
        score_factor=0.25,
        seed=0,
    )
