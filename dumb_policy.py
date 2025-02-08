import numpy as np
import random
import pygame
import time

from snake import SnakeEnv

def idx2pos(idx, dim):

    x = idx[1] + 1
    y = dim - idx[0]

    return [x, y]

def pos2idx(pos, dim):

    i = dim - pos[1]
    j = pos[0] - 1

    return [i, j]


def dumb_policy(state, dim):

    actions = ['RIGHT', 'LEFT', 'UP', 'DOWN']

    head_idx = np.argwhere(state[0,:,:])[0]
    fruit_idx = np.argwhere(state[2,:,:])[0]

    head_pos = idx2pos(head_idx, dim)
    fruit_pos = idx2pos(fruit_idx, dim)

    
    
    # If fruit above, can move up, go up
    if fruit_pos[1] > head_pos[1] and state[1, head_idx[0] - 1, head_idx[1]] == 0:
        return "UP"

    # if fruit right, can move right, go right
    if fruit_pos[0] > head_pos[0] and state[1, head_idx[0], head_idx[1] + 1] == 0:
        return "RIGHT"

    # if fruit left, can move left, go left
    if fruit_pos[0] < head_pos[0] and state[1, head_idx[0], head_idx[1] - 1] == 0:
        return "LEFT"

    # if fruit down, can go down, go down
    if fruit_pos[1] < head_pos[1] and state[1, head_idx[0] + 1, head_idx[1]] == 0:
        return "DOWN"


    return "UP"


def eval(n, dim=10):

    scores = []

    for _ in range(n):
        env = SnakeEnv(dim=5, headless=False)

        alive = True
        game_over = False

        while alive and not game_over:
            state = env.get_state()
            dir = dumb_policy(state, env.dim)
            # dir = random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
            game_over, alive, reward, next_state = env.step_env(dir, print_board=False)
            
            if game_over or not alive:
                scores.append(env.score)
            # time.sleep(0.1)
    
    print(sum(scores) / len(scores))

eval(100)