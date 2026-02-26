import pickle
from time import sleep
from hockey.hockey_env import HockeyEnv

def visualize_game(game_data):
    env = HockeyEnv()

    for round in game_data["rounds"]:
        observations = round["observations"]
        for obs in observations:
            env.set_state(obs)
            env.render()
            sleep(0.05)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to game file")

    args = parser.parse_args()
    
    with open(args.file, "rb") as f:
        data = pickle.load(f)

    visualize_game(data)