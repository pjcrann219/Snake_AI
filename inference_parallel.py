from snake import SnakeEnv
from snake_multi import SnakeEnvParallel
from policies import *
from utils import *
import torch



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

GAME_DIM = 7

a2c = A2C_10_Features(state_dim=10, beta=0, gamma=1, device=device, 
               lr_actor=0, actor_step_size = 1, actor_gamma = 1, min_lr_actor = 1e-3,
               lr_critic=0, critic_step_size = 1, critic_gamma = 1, min_lr_critic = 1e-3)
# a2c.load_state_dict(torch.load('models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth', map_location=device))

snake_manager = SnakeEnvParallel(a2c, 2, dim=GAME_DIM, headless=False, device=device, rewards=[0, 0, 0, 0])
snake_manager.brain.load_state_dict(torch.load('saved_models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth', map_location=device))

for epoch in range(10):
    snake_manager.reset_envs(display_example=True)
    average_score, average_game_length, win_percent, max_score, runtime = snake_manager.inference_envs(plot_activation=False)
