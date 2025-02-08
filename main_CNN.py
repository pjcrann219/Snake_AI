import numpy as np
import random
import pygame
import time
import torch

from snake import SnakeEnv
from policy import A2C_CNN

GAME_DIM = 16

scores = []
num_epochs = 10000

show_every_n_games = 100


a2c = A2C_CNN(state_dim=GAME_DIM, lr_actor=1e-3, lr_critic=1e-3)

# get experience in this environment
for epoch in range(num_epochs):
    if (epoch) % show_every_n_games == 0:
        headless = False
    else:
        headless = True
    env = SnakeEnv(dim=GAME_DIM, headless=headless, rewards=[1, -1, -0.01, 0.02])

    alive = True
    game_over = False

    while alive and not game_over:
        # Get state
        state = env.get_state()
        state_torch = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get action, log_prob, and value from a2c
        action, log_prob, entropy, value = a2c.get_action(state_torch)

        # Step envirnment according to action
        game_over, alive, reward, next_state = env.step_env(action, print_board=False)

        # print(f"Game Over: {game_over}, Alive: {alive}, Reward: {reward}, Next State: {next_state}")
        # print()


        # Calculate next value for TD target
        if not alive or game_over:
            done = True
            next_value = torch.tensor([[0]])
        else:
            done = False
            next_state_torch = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_value = a2c.get_value(next_state_torch, nograd=True) # Do we need no grad if we use td_target.detach() below?

        # Compute advantage function
        td_target = reward + a2c.gamma * next_value * (1 - done)
        advantage = td_target - value

        # Update critic/actor loss
        a2c.critic_loss = torch.nn.functional.mse_loss(value, td_target.detach())
        a2c.actor_loss = -log_prob * advantage.detach() - a2c.beta * entropy
        
        a2c.critic_optimizer.zero_grad()
        a2c.actor_optimizer.zero_grad()

        total_loss = a2c.critic_loss + a2c.actor_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(a2c.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(a2c.critic.parameters(), max_norm=0.5)

        a2c.critic_optimizer.step()
        a2c.actor_optimizer.step()

        if game_over or not alive:
            scores.append(env.score)

        if not headless:
            time.sleep(0.2)
    
    print(f"Game length: {env.game_length}, Score: {env.score}")

print(sum(scores) / len(scores))
print(sum(scores))
print(len(scores))
