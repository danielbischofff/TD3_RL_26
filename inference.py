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
        print(f"game ended: {text_result} | my score: {stats[0]} vs opp score: {stats[1]}")


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
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} | my score: {stats[0]} vs opp score: {stats[1]}")


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
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} | my score: {stats[0]} vs opp score: {stats[1]}")


# -------------------------
# Local evaluation runner
# -------------------------

def run_local_games_vs_weak(
    agent: Agent,
    n_games: int = 20,
    video_path: str = "hockey_vs_weak.mp4",
    record_games: int = 1,          # how many episodes to record into the video (1 keeps file small)
    fps: int = 50,
    score_factor: float = 0.25,
    seed: int = 0,
):
    """
    Runs games locally vs BasicOpponent(weak=True) using HockeyEnv_BasicOpponent.
    Records up to `record_games` episodes into an MP4 and prints win/loss stats + score.
    """
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    # Single-agent wrapper env: you supply 4D action; opponent is built-in.
    env = h_env.HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=True)

    rng = np.random.default_rng(seed)

    wins = losses = ties = 0
    frames = []

    for ep in range(n_games):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

        # "game_id" in the guide is bytes; for local eval we can synthesize one:
        fake_game_id = uuid.uuid4().int.to_bytes(16, byteorder="big", signed=False)
        agent.on_start_game(fake_game_id)

        terminated = truncated = False
        ep_return = 0.0

        # record initial frame
        if ep < record_games:
            frames.append(env.render(mode="rgb_array"))

        t = 0
        while True:
            action = agent.get_step(obs.tolist())   # agent expects list[float]
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))

            ep_return += float(reward)

            if ep < record_games:
                frames.append(env.render(mode="rgb_array"))

            t += 1
            if terminated or truncated:
                break

        winner = info.get("winner", 0)  # 1: agent won, -1: opponent won, 0: tie
        if winner == 1:
            wins += 1
            result_bool = True
        elif winner == -1:
            losses += 1
            result_bool = False
        else:
            ties += 1
            result_bool = False  # ties counted as not-wins in this boolean

        # Stats format like the guide prints (my score vs opp score).
        # If you want "score" to be episode return, use ep_return; otherwise just use winner-based.
        stats = [float(ep_return), 0.0]
        agent.on_end_game(result_bool, stats)

        print(f"[EP {ep:03d}] steps={t} | winner={winner} | return={ep_return:.3f}")

    env.close()

    # Write video
    if frames:
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"\nSaved video to: {video_path}")

    # Print summary score
    total = wins + losses + ties
    score = (wins - losses) * score_factor

    print("\n=== Summary ===")
    print(f"Games: {total}")
    print(f"Wins / Losses / Ties: {wins} / {losses} / {ties}")
    print(f"Score = (wins - losses) * factor = ({wins} - {losses}) * {score_factor} = {score:.3f}")


if __name__ == "__main__":
    # Choose one:
    # agent = RandomAgent()
    agent = HockeyAgent_TD3(ckpt_path="/TD3_RL_26/checkpoints")

    run_local_games_vs_weak(
        agent=agent,
        n_games=20,
        video_path="hockey_vs_weak.mp4",
        record_games=1,
        fps=50,
        score_factor=0.25,
        seed=0,
    )
