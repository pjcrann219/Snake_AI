from snake import SnakeEnv
from snake_multi import SnakeEnvParallel
from policies import *
from utils import *

import torch
import os
import glob
import json
import datetime
from torch.utils.tensorboard import SummaryWriter

params = {
    'num_env':  1000,
    'lr_actor': 3e-3,
    'lr_critic': 5e-3,
    'actor_step_size': 20,
    'critic_step_size': 20,
    'actor_gamma': 0.80,
    'critic_gamma': 0.80,
    'min_lr_actor': 2e-3,
    'min_lr_critic': 2e-3,
    'beta': 0.1,
    'gamma': 0.98,
    'rewards': [1, -1, 0.01, 0],
    'num_epochs': 1000,
    'GAME_DIM': 10
}

experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(experiment_name, '\t', json.dumps(params, separators=(", ", ": "), sort_keys=False))

writer = SummaryWriter(f"runs/feat_{experiment_name}")
for key, value in params.items():
    if isinstance(value, (int, float)):
        writer.add_scalar(f"hyperparams/{key}", value, 0)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            writer.add_scalar(f"hyperparams/{key}_{i}", v, 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

GAME_DIM = params['GAME_DIM']

a2c = A2C_10_Features(state_dim=10, beta=params['beta'], gamma=params['gamma'], device=device, 
               lr_actor=params['lr_actor'], actor_step_size = params['actor_step_size'], actor_gamma = params['actor_gamma'], min_lr_actor = params['min_lr_actor'],
               lr_critic=params['lr_critic'], critic_step_size = params['critic_step_size'], critic_gamma = params['critic_gamma'], min_lr_critic = params['min_lr_critic'])

snake_manager = SnakeEnvParallel(a2c, params['num_env'], dim=GAME_DIM, headless=True, device=device, rewards=params['rewards'])
snake_manager.brain.load_state_dict(torch.load('models/snake_a2c_Feat_dim10_epoch990_20250218-171356.pth', map_location=device))

for epoch in range(params['num_epochs']):
    average_score, average_game_length, win_percent, max_score, runtime = snake_manager.train_envs()
    if epoch%10 == 0:
        snake_manager.reset_envs(display_example=True)
    else:
        snake_manager.reset_envs(display_example=False)

    print(f"{epoch}: Avg Score: {average_score:0.3f},\tAvg Game Len: {average_game_length:0.3f},\truntime: {runtime:.6f},\tMax Score: {max_score},\tWin %: {win_percent},\tActor LR: {snake_manager.brain.actor_scheduler.get_last_lr()[0]:.2e}")
    
    writer.add_scalar("Avg Score", average_score, epoch)
    writer.add_scalar("Avg Game Length", average_game_length, epoch)
    writer.add_scalar("Max Score", max_score, epoch)
    writer.add_scalar("Win Percent", win_percent, epoch)
    writer.add_scalar("Actor LR", snake_manager.brain.actor_scheduler.get_last_lr()[0], epoch)
    writer.add_scalar("Critic LR", snake_manager.brain.critic_scheduler.get_last_lr()[0], epoch)


    # Delete past safe for this run
    if epoch%10 == 0:
        torch.save(snake_manager.brain.state_dict(), f'models/snake_a2c_Feat_dim{GAME_DIM}_epoch{epoch}_{experiment_name}.pth')
        # Delete past safe for this run
        for file in glob.glob(f"models/snake_a2c_Feat_dim{GAME_DIM}_epoch*_{experiment_name}.pth"):
            if f"_epoch{epoch}_" not in file:  # Keep the current epoch
                os.remove(file)
                print(f"Deleted: {file}")