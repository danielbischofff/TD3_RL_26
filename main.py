from td3 import TD3_trainer, TD3_agent
import torch
import torch.optim as optim
import torch.nn.functional as F
import hockey.hockey_env as h_env
import numpy as np
import wandb


# ---------------
# Initialization
# ---------------

# --- Env init ---
env = h_env.HockeyEnv() 
env.reset()  # Reset environment to initialize observation_space
obs2 = env.obs_agent_two() # initiate two agents
obs_dim = env.observation_space.shape[0]
act_dim = env.num_actions
act_bounds = (env.action_space.low[0], env.action_space.high[0])
resume = "/home/stud359/TD3_RL_26/checkpoints/td3_ckp_04_so.pt"

# --- model init ---
td3_trainer = TD3_trainer(obs_dim, act_dim, act_bounds, resume)
batch_size = td3_trainer.config["batch_size"]
policy_delay = 2
device = td3_trainer.device
total_it = 0
max_timesteps = 600

# --- opponent init ---
opponent = "td3"

# import os
# os.environ["WANDB_MODE"] = "disabled"

# --- logging init ---
run = wandb.init(
    entity="bischoffd",
    name="td3_run_005",
    project="RL_TD3_hockey",
    config=td3_trainer.config,
) 

# ----------
# Train run
# ----------

for eps in range(td3_trainer.max_episodes):

    # --- Create Opponent ---
    if opponent == "strong":
        player2 = h_env.BasicOpponent(weak=False)
    if opponent == "td3":
        player2 = TD3_agent(obs_dim=obs_dim, act_dim=act_dim,act_bounds=act_bounds, ckpt_path="checkpoints/td3_ckp_04_so.pt")
    else:
        player2 = h_env.BasicOpponent(weak=True)

    # --- Env Step ---
    obs1, info = env.reset()
    obs2 = env.obs_agent_two()
    print(f"episode {eps}")
    episode_return = 0
    t = 0
    critic1_losses = []
    critic2_losses = []
    actor_losses = []
    
    # while not terminated or truncated:
    for t in range(max_timesteps):

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
        # r = + info["reward_puck_direction"]+ info["reward_touch_puck"] 

        episode_return += r
        total_it += 1

        # --- Add to Buffer ---
        td3_trainer.buffer.add(obs1, a1, r, obs1_new, terminated)

        obs1 = obs1_new # make new obs to current
        obs2 = env.obs_agent_two()

        # ---- LEARNING PHASE ----
        if total_it >= td3_trainer.config["warm_up"] and td3_trainer.buffer.size >= batch_size:

            states, actions, rewards, next_states, dones_b = td3_trainer.buffer.sample(batch_size)

            state_b = torch.FloatTensor(states).to(device)
            action_b = torch.FloatTensor(actions).to(device)
            reward_b = torch.FloatTensor(rewards).to(device)
            next_state_b = torch.FloatTensor(next_states).to(device)
            dones_b = torch.from_numpy(dones_b.astype(np.float32)).to(device)
            dones_b = torch.as_tensor(dones_b, dtype=torch.float32, device=device)

            assert state_b.shape == (batch_size, obs_dim)
            assert action_b.shape == (batch_size, act_dim)
            assert reward_b.shape == (batch_size, 1)
            assert dones_b.shape == (batch_size, 1)
            assert state_b.device.type == device.type
            assert td3_trainer.actor.l1.weight.device.type == device.type


            # ---- CRITIC UPDATE ----
            critic_loss = td3_trainer.critic_update(state_b, action_b, reward_b, next_state_b, dones_b)
            # run.log({"critic_1_loss": critic_loss["critic_1_loss"], "critic_2_loss": critic_loss["critic_2_loss"]}, step=total_it)

            critic1_losses.append(critic_loss["critic_1_loss"])
            critic2_losses.append(critic_loss["critic_2_loss"])

            # ---- ACTOR UPDATE ----
            if total_it % policy_delay == 0:
                agent_loss = td3_trainer.actor_target_update(state_b)
                # run.log({"agent_loss":agent_loss.item()}, step=total_it)
                actor_losses.append(agent_loss.item())


        if terminated or truncated:
            break

    # --- CHECKPOINTING per step ---
    if eps > 0 and eps % td3_trainer.config["checkpoint_interval"] == 0:
        td3_trainer.save_checkpoint(step=str(total_it))

    run.log({
    "episode_return": episode_return,
    "episode_length": t + 1,
    "winner": info.get("winner", None),

    "critic_1_loss": float(np.mean(critic1_losses)) if critic1_losses else None,
    "critic_2_loss": float(np.mean(critic2_losses)) if critic2_losses else None,
    "agent_loss": float(np.mean(actor_losses)) if actor_losses else None,
    }, step=total_it)

td3_trainer.save_checkpoint(step="last")