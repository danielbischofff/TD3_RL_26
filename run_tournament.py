from __future__ import annotations

import argparse
import uuid

import numpy as np

import hockey.hockey_env as h_env
from td3 import TD3_agent
from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class HockeyAgent_TD3(Agent):

    def __init__(self, ckpt_path) -> None:
        super().__init__()
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.num_actions
        act_bounds = (env.action_space.low[0], env.action_space.high[0])
        self.hockey_agent = TD3_agent(obs_dim=obs_dim, act_dim=act_dim,act_bounds=act_bounds, ckpt_path=ckpt_path)

    def get_step(self, observation: list[float]) -> list[float]:

        action = self.hockey_agent.act(observation).tolist()
        # action = np.concatenate([action, [0.9,0,0,0]])
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["mixed04", "so04"],
        default="td3",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "so04":
        agent = HockeyAgent_TD3(ckpt_path = "/Users/danielbischoff/Documents/MasterInformatik/RL/FinalProject/checkpoints/td3_ckp_04_so.pt")
    elif args.agent == "mixed04":
        agent = HockeyAgent_TD3(ckpt_path = "/Users/danielbischoff/Documents/MasterInformatik/RL/FinalProject/checkpoints/td3_ckp_mixed_04.pt")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
