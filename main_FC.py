import numpy as np
import random
import pygame
import time
import torch

from snake import SnakeEnv
from policy import A2C_FC

GAME_DIM = 5

scores = []
num_epochs = 100000

show_every_n_games = 1000

last_n_scores = []
last_n_lengths = []

a2c = A2C_FC(state_dim=GAME_DIM, lr_actor=5e-3, lr_critic=5e-3)

# get experience in this environment
for epoch in range(num_epochs):
    if (epoch) % show_every_n_games == 0:
        headless = False
    else:
        headless = True
    env = SnakeEnv(dim=GAME_DIM, headless=headless, rewards=[1, -1, -0.01, 0])

    done = False

    while not done:
        # Get state
        state = env.get_state()
        state = state[0:3,:,:]
        state_torch = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get action, log_prob, and value from a2c
        action, log_prob, entropy, value = a2c.get_action(state_torch)

        # Step envirnment according to action
        game_over, alive, reward, next_state = env.step_env(action, print_board=False)
        if next_state is not None:
            next_state = next_state[0:3,:,:]

        # print(f"Game Over: {game_over}, Alive: {alive}, Reward: {reward}, Next State: {next_state}")

        # determine if game is over
        if not alive or game_over:
            done = True
        else:
            done = False

        # Update loss and optimize
        a2c.update_loss(entropy, log_prob, value, next_state, reward, done, optimize=True)

        if not headless:
            time.sleep(0.2)
    
    scores.append(env.score)
    last_n_scores.append(env.score)
    last_n_lengths.append(env.game_length)

    if not headless:
        print(f"Epoch: {epoch}, average score: {sum(last_n_scores) / len(last_n_scores):0.3f}, average game length: {sum(last_n_lengths) / len(last_n_lengths):0.3f}")
        last_n_scores = []

    # print(f"Game length: {env.game_length}, Score: {env.score}")

print(sum(scores) / len(scores))
print(sum(scores))
print(len(scores))
