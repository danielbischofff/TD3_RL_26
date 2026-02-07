from td3_trainer import TD3_trainer
import torch
import torch.optim as optim
import torch.nn.functional as F
import hockey.hockey_env as h_env
import numpy as np
import wandb

# ---------------
# Initialization
# ---------------

env = h_env.HockeyEnv() 
obs2 = env.obs_agent_two() # initiate two agents

td3_trainer = TD3_trainer(env)
batch_size = td3_trainer.config["batch_size"]
policy_delay = 2
device = td3_trainer.device
total_it = 0

# import os
# os.environ["WANDB_MODE"] = "disabled"

# WANDB logging
run = wandb.init(
    entity="bischoffd",
    name="td3_run_002",
    project="RL_TD3_hockey",
    config=td3_trainer.config,
) 

# ----------
# Train run
# ----------

for eps in range(td3_trainer.max_episodes):
    obs1, info = env.reset()
    obs2 = env.obs_agent_two()
    player2 = h_env.BasicOpponent(weak=True)
    print(f"episode {eps}")

    done = False
    episode_return = 0
    
    while not done:
        # ---- Buffer Step ----
        if total_it < td3_trainer.config["warm_up"]:
            # Warmup action
            a1 = td3_trainer.select_random_action()
        else:
            # Policy action
            a1 = td3_trainer.select_action_with_policy(obs1)

        if total_it == td3_trainer.config["warm_up"]: # warmup finished logging
            run.log({"event/warmup_finished": 1}, step=total_it)

        a2 = player2.act(obs2)

        # --- Env step ---
        obs1_new, r, terminated, truncated, info = env.step(np.hstack([a1, a2]))
        done = terminated or truncated
        episode_return += r
        total_it += 1

        # --- Add to Buffer ---
        td3_trainer.buffer.add(obs1, a1, r, obs1_new, done)

        # ---- LEARNING PHASE ----
        if total_it >= td3_trainer.config["warm_up"] and td3_trainer.buffer.size >= batch_size:

            state, action, reward, next_state, done_b = td3_trainer.buffer.sample(batch_size)

            state_b = torch.FloatTensor(state).to(device)
            action_b = torch.FloatTensor(action).to(device)
            reward_b = torch.FloatTensor(reward).to(device)
            next_state_b = torch.FloatTensor(next_state).to(device)
            done_b = torch.from_numpy(done_b.astype(np.float32))
            
            # ---- CRITIC UPDATE ----
            critic_loss = td3_trainer.critic_update(state_b, action_b, reward_b, next_state_b, done_b)
            run.log({"critic_loss": critic_loss["critic_loss"], "critic_1_loss": critic_loss["critic_1_loss"], "critic_2_loss": critic_loss["critic_2_loss"]}, step=total_it)

            # ---- ACTOR UPDATE ----
            if eps % policy_delay == 0:
                agent_loss = td3_trainer.actor_target_update(state_b)
                run.log({"agent_loss":agent_loss}, step=total_it)

            # --- CHECKPOINTING per step ---
            if eps > 0 and eps % td3_trainer.config["checkpoint_interval"] == 0:
                td3_trainer.save_checkpoint(step=str(total_it))

    run.log({"episode_return": episode_return}, step=total_it)


td3_trainer.save_checkpoint(step="last")