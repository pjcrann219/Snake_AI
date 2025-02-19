from snake_multi import SnakeEnvParallel
from policies import *

def run_test(snake_manager):
    for epoch in range(10):
        snake_manager.reset_envs(display_example=True)
        average_score, average_game_length, win_percent, max_score, runtime = snake_manager.inference_envs(plot_activation=False)


def load_test(test='t1', headless=False, num_envs = 1):
    # 7x7 test with 10 features [x from fruit, y from fruit, dist wall up, dist wall down, dist wall left, dist wall right, ...
    # dist body up, dist body down, dist body left, dist body right] / GAME_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = 't1'
    if test == 't1':
        GAME_DIM = 7
        state_dim = 10

        print(f"Test 1 loaded with Game Dim: {GAME_DIM}, State Dim: {10}, Num Envs: {num_envs}, headless: {headless}")

        a2c = A2C_10_Features(state_dim=10, beta=0, gamma=1, device=device, 
                lr_actor=0, actor_step_size = 1, actor_gamma = 1, min_lr_actor = 1e-3,
                lr_critic=0, critic_step_size = 1, critic_gamma = 1, min_lr_critic = 1e-3)
        # a2c.load_state_dict(torch.load('models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth', map_location=device))

        snake_manager = SnakeEnvParallel(a2c, num_envs, dim=GAME_DIM, headless=headless, device=device, rewards=[0, 0, 0, 0])
        snake_manager.brain.load_state_dict(torch.load('saved_models/snake_a2c_Feat_dim7_epoch690_20250215-134600.pth', map_location=device))

        return snake_manager, device

    return None, None

snake_manager, device = load_test(test='t1', headless=False, num_envs = 1)
run_test(snake_manager=snake_manager)