# Project Notes

## Question

- How much code do we have to code by our self (libaries etc.)?


## Todos

- [ ] implement algo TD3
- [ ] understand env hockey player
- [ ] understand Cluster building

## TD3

### DDPG (Deep Deterministic Policy Gradient)

model-free off policy algorithm for continuous action spaces

- **model-free:** model doesn't need to know the environment to predict outcomes
- **Off-policy:** DDPG can learn from actions outside of its current policy, which means it can explore the environment more efficiently

- **Actor-Critic Architecture:** Actor suggest actions, critic evaluates how good those actions are
- **Replay Buffer:** stores past experiences so the agent can learn from them repeatedly
- **Target Networks:** slightly delayed copies of the main networks (actor and critic), which help stabilize the training process.

#### Problems of DDPG

1. **Overestimation Bias:** The critic oftern overestimates the action values, leading to suboptimal decisions. Over time this adds up.
2. **Sensitivity to Hyperparameters:** DDPG is highly sensitive to learning rate and noise parameters
3. **Training Instability:** unstable during training, especially in complex environments.

-> Twin Delayed Deep Deterministic Policy Gradient (TD3)

### Twin Delayed Deep Deterministic Policy Gradient (TD3)

Still model-free, off-policy algo + clever tricks to stabilize learning and improve performance

#### Core Innovations of TD3

1. Double Critics (Twin Critics): instead of one
2. Delayed Policy Updates: policy is updated less freweuntly than critics -> more stable learning
3. Target Policy Smoothing: adding noise to the target actions, TD3 avoids overfitting to sharp, narrow peaks in the Q-value function resulting in smoother and more reliable policies

#### Double Critics (Twin Critics)

- Ask to critics and take the more conservative answear
- two critics networks to estimate the value of actions
- take minimum Q-value between the two critics
-> reduce overestimation bias

#### Delayed Policy Updates

- delay actors update relative to the critics
- to correct for instability
- ensure relaible, stable Q-values estimates

#### Target Policy Smoothing

- add noise to critic's update
-> prevents Q-function from overfitting to small, sharp peaks in the value function

#### Pseudocode in TD3

````python
    Initialize actor network (π) and two critic networks (Q₁, Q₂)
    Initialize target networks (π', Q₁', Q₂') as copies of the original networks
    Initialize replay buffer
````

````python

    for each timestep:
    Select action a = π(s) + noise
    Execute action a in the environment, observe reward r, and next state s'
    Store (s, a, r, s) in the replay buffer    if timestep > update_start:
        for each update_step:
            Sample a minibatch of transitions (s, a, r, s') from the replay buffer
            Compute target actions: a' = π'(s') + clipped_noise
            Compute target Q-value: y = r + γ * min(Q₁'(s', a'), Q₂'(s', a'))
            
            Update critics: minimize (Q₁(s, a) - y)² and (Q₂(s, a) - y)²
            
            if update_step % delay == 0:
                Update the actor using the policy gradient
                Update target networks: 
                    π' ← τ * π + (1 - τ) * π'
                    Q₁' ← τ * Q₁ + (1 - τ) * Q₁'
                    Q₂' ← τ * Q₂ + (1 - τ) * Q₂'
````

#### loss functions

- Critic loss: MSE between predicted Q-value and Bellman target
- Actor loss:
    Negative Q-value of the actor’s actions

- Critics updated every step
- Actor updated every N steps (delayed)
- Two critics → two separate losses

## Hockey Env

### States / Oberservations

````html
# 0  x pos player one
# 1  y pos player one
# 2  angle player one
# 3  x vel player one
# 4  y vel player one
# 5  angular vel player one
# 6  x player two
# 7  y player two
# 8  angle player two
# 9 y vel player two
# 10 y vel player two
# 11 angular vel player two
# 12 x pos puck
# 13 y pos puck
# 14 x vel puck
# 15 y vel puck
# Keep Puck Mode
# 16 time left player has puck
# 17 time left other player has puck
````

#### Ablauf TD3

1. Warm-Up:
    - Random action
    - env.step
    - store transitions

2. Learning Phase:

    2.1 ACT:
        - action = actor(state) + exploration noise
        - env.step
        - store transitions

    2.2 CRITIC UPDATE (every step):
        - sample batch
        - target_action = actor_target(next_states) + noise
        - target_Q = reward + gamma * min(Q1_target, Q2_target)
        - critic_loss = MSE(Q1) + MSE(Q2)
        - update critic_1 and critic_2 via gradient descent

    2.3 ACTOR UPDATE (every policy_delay steps):
        - actor_loss = -Q1(state, actor(state))
        - update actor via gradient descent
        - SOFT UPDATE TARGET NETWORKS:
            actor_target
            critic_1_target
            critic_2_target

## Next Todos

- [x] train for some episode and see if it works
- [x] add logging
- [ ] add inference stuff

registraion key: rl-hockey-26

> 07-02-2026

Where are we at today?
- does my code work 
- what is missing
- inference would be a good check

I could do inference and with this have a look how good i really plays. This could help me understand what if it is working. A good result would be if my trained model trains better than a random model. This could also be a little hard.
it also would be important to see where i can run my model on the cluster. Set up the script etc. Problems could be that i do not have a vpn connection, which i probably need. I could ask daniel.

1. Check cluster thing for training
    - [x] Upload project to git
    - [ ] download it to cluster (in VPN)
    - [ ] make singularity docker container (from container file)

2. Do inference
    - [x] understand how inference would work at the end
    - [x] added inference

3. Do training

4. Check for bugs
    - [ ] could try to train on smaller problem
    - [ ] check with ChatGPT what could be wrong
    - [ ] check on easiere env

> 09-02-2026

## Save

- Rewardkurve für pro episode das training gegen verschiedene Opponents
- Actor loss / Critic loss
- episoden längen

## train

- weak opp
- strong opp
- gegen alte opp
- Gegeneinander

## Ablauf

09.02

- Introduction

10.02

- fertig werden mit Coden
- Nathalie corregiert Introduction