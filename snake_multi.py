from snake import SnakeEnv
from policies import *
from utils import *

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


def setup_visualization(features, activations, layer_names, sizes=[300, 20, 20, 20, 300, 20, 300]):
    """Set up initial visualization"""
    # Convert features to numpy and add to activations
    features_np = features.detach().cpu().numpy()
    all_activations = {'input': features_np}
    all_activations.update(activations)
    
    # Create figure
    fig, axes = plt.subplots(len(all_activations), 1, figsize=(15, 2*len(all_activations)))
    if len(all_activations) == 1:
        axes = [axes]
    
    scatter_plots = []
    # Create scatter plots with data
    for ax, ((layer, activation), layer_name, size) in zip(axes, zip(all_activations.items(), layer_names, sizes)):
        act = activation[0].flatten() if layer != 'input' else activation.flatten()
        x = np.arange(len(act))
        y = np.zeros_like(x)
        
        scatter = ax.scatter(x, y, c=act, cmap=plt.cm.RdBu, s=size, 
                           vmin=-1, vmax=1, edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax)
        scatter_plots.append(scatter)
        
        ax.set_title(layer_name, fontsize=12, pad=20, fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, len(act))
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes, scatter_plots

def update_visualization(features, activations, fig, axes, scatter_plots):
    """Update existing visualization"""
    # Convert features to numpy and add to activations
    features_np = features.detach().cpu().numpy()
    all_activations = {'input': features_np}
    all_activations.update(activations)
    
    # Update each scatter plot
    for scatter, (layer, activation) in zip(scatter_plots, all_activations.items()):
        act = activation[0].flatten() if layer != 'input' else activation.flatten()
        x = np.arange(len(act))
        y = np.zeros_like(x)
        
        scatter.set_offsets(np.c_[x, y])
        scatter.set_array(act)
    
    fig.canvas.draw_idle()
    plt.pause(0.1)

# # Register hooks
# activations = {}
# handles = []

# def hook_fn(module, input, output):
#     activations[module] = output.detach().cpu().numpy()


STATE_DIM = 10

class SnakeEnvParallel():
    def __init__(self, brain, num_envs, dim=5, headless=True, rewards=[1, -1, 0.1, 0], device=torch.device('cpu')):
        self.device = device

        self.num_envs = num_envs
        self.dim = dim
        self.rewards = rewards

        self.brain = brain
        self.replay = None
        self.max_score = 0

        self.reset_envs()

        # # Define layers and names
        # self.layers_of_interest = [
        #     self.brain.features[2], self.brain.features[4],
        #     self.brain.actor[2], self.brain.actor[4],
        #     self.brain.critic[2], self.brain.critic[3]
        # ]

        # self.names = [
        #     "Input",
        #     "Feature Layer 1", "Feature Layer 2",
        #     "Actor Layer 1", "Action",
        #     "Critic Layer 1", "Value"
        # ]
    
    def reset_envs(self, display_example=False):
        if display_example:
            self.envs = [SnakeEnv(dim=self.dim, headless=False, rewards=self.rewards)]
            for _ in range(self.num_envs-1):
                self.envs.append(SnakeEnv(dim=self.dim, headless=True, rewards=self.rewards))
        else:
            self.envs = [SnakeEnv(dim=self.dim, headless=True, rewards=self.rewards) for _ in range(self.num_envs)]

        self.dones = [False for _ in range(self.num_envs)]
        self.update_active_env_idx()

    def set_replay(self, replay):
        self.replay = replay

    def train_envs_PPO(self):
        start_time = time.time()
        i = 0
        while False in self.dones:
            features = self.get_features()
            actions, log_probs, entropies, values = self.get_actions(features)
            next_features_torch, rewards, finished = self.step_envs(actions)

            self.compute_loss(next_features_torch, rewards, values, log_probs, entropies, finished)
            self.optimize()
            self.update_active_env_idx()
            # print(f"{i}:, features: {features.shape}, next features: {next_features_torch.shape}, num_dones: {sum(snake_manager.dones)}, num_finished: {sum(finished)}")
            i += 1
            if not self.dones[0]:
                time.sleep(0.2)

        # Step scheudlers
        self.brain.actor_scheduler.step()
        for param_group in self.brain.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.brain.min_lr_actor)  # Set min LR for actor

        self.brain.critic_scheduler.step()
        for param_group in self.brain.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.brain.min_lr_critic)  # Set min LR for critic
        
        scores = [env.score for env in self.envs]
        average_score = sum(scores) / len(scores)
        max_score = max(scores)
        if max_score > self.max_score:
            self.max_score = max_score

        win_percent = sum([1 for score in scores if score == self.dim **2]) / len(scores)

        game_lengths = [env.game_length for env in self.envs]
        average_game_length = sum(game_lengths) / len(game_lengths)

        runtime = time.time() - start_time

        return average_score, average_game_length, win_percent, self.max_score, runtime

    def train_envs(self):
        start_time = time.time()
        i = 0
        while False in self.dones:
            features = self.get_features()
            actions, log_probs, entropies, values = self.get_actions(features)
            next_features_torch, rewards, finished = self.step_envs(actions)
            self.compute_loss(next_features_torch, rewards, values, log_probs, entropies, finished)
            self.optimize()
            self.update_active_env_idx()
            # print(f"{i}:, features: {features.shape}, next features: {next_features_torch.shape}, num_dones: {sum(snake_manager.dones)}, num_finished: {sum(finished)}")
            i += 1
            if not self.dones[0]:
                time.sleep(0.2)

        # Step scheudlers
        self.brain.actor_scheduler.step()
        for param_group in self.brain.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.brain.min_lr_actor)  # Set min LR for actor

        self.brain.critic_scheduler.step()
        for param_group in self.brain.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.brain.min_lr_critic)  # Set min LR for critic
        
        scores = [env.score for env in self.envs]
        average_score = sum(scores) / len(scores)
        max_score = max(scores)
        if max_score > self.max_score:
            self.max_score = max_score

        win_percent = sum([1 for score in scores if score == self.dim **2]) / len(scores)

        game_lengths = [env.game_length for env in self.envs]
        average_game_length = sum(game_lengths) / len(game_lengths)

        runtime = time.time() - start_time

        return average_score, average_game_length, win_percent, self.max_score, runtime
    
    def inference_envs(self, plot_activation=False):
        if plot_activation:
            # Register hooks
            activations = {}
            handles = []

            def hook_fn(module, input, output):
                activations[module] = output.detach().cpu().numpy()

            for layer in self.layers_of_interest:
                handles.append(layer.register_forward_hook(hook_fn))

        start_time = time.time()
        i = 0
        while False in self.dones:

            features = self.get_features()
            actions, _, _, _ = self.get_actions(features)
            next_features_torch, _, finished = self.step_envs(actions)

            self.update_active_env_idx()
            if plot_activation:
                if i == 0:
                    fig, axes, scatter_plots = setup_visualization(features, activations, self.names)
                else:
                    update_visualization(features, activations, fig, axes, scatter_plots)

            # print(f"{i}:, features: {features.shape}, next features: {next_features_torch.shape}, num_dones: {sum(snake_manager.dones)}, num_finished: {sum(finished)}")
            i += 1
            if not self.dones[0]:
                time.sleep(0.2)
        
        scores = [env.score for env in self.envs]
        average_score = sum(scores) / len(scores)
        max_score = max(scores)
        if max_score > self.max_score:
            self.max_score = max_score

        win_percent = sum([1 for score in scores if score == self.dim **2]) / len(scores)

        game_lengths = [env.game_length for env in self.envs]
        average_game_length = sum(game_lengths) / len(game_lengths)

        runtime = time.time() - start_time

        return average_score, average_game_length, win_percent, self.max_score, runtime
    
    def update_active_env_idx(self):
        # Set active env indecies (valid if done == True)
        self.active_env_idx = [i for i, done in enumerate(self.dones) if not done]

    def get_features(self):
        # Return tensor of all valid states
        features_tensors = torch.zeros((len(self.active_env_idx), STATE_DIM), dtype=torch.float32, device=self.device)
        for i, idx in enumerate(self.active_env_idx):
            env = self.envs[idx]
            state = env.get_state()
            features = get_features_10(state, env.dim)
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
            features_tensors[i] = features
        
        return features_tensors
    
    def get_actions(self, features: torch.tensor):
        actions, log_probs, entropies, values = self.brain.get_action(features)
        return actions, log_probs, entropies, values
    
    def step_envs(self, actions):
        rewards = []
        finished = []
        next_features_torch = torch.zeros((len(self.active_env_idx), STATE_DIM), device=self.device)
        for i, idx in enumerate(self.active_env_idx):
            env = self.envs[idx]
            game_over, alive, reward, next_state = env.step_env(int(actions[i]), print_board=False)

            if next_state is not None:
                next_features = torch.tensor(get_features_10(next_state, env.dim), dtype=torch.float32, device=self.device)
            else:
                next_features = torch.zeros(STATE_DIM, dtype=torch.float32, device=self.device)

            # determine if game is over
            if not alive or game_over:
                self.dones[idx] = True

            rewards.append(reward)
            finished.append(self.dones[idx])
            next_features_torch[i] = next_features

        return next_features_torch, rewards, finished
    
    def compute_loss(self, next_features_torch, rewards, values, log_probs, entropies, finished):
        next_values = self.brain.get_value(next_features_torch, nograd=True)

        done_tensor = torch.tensor(finished, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        td_targets = rewards_tensor + self.brain.gamma * (next_values.T * (1-done_tensor))
        advantages = td_targets - values.detach().T
        
        self.brain.critic_loss = torch.nn.functional.mse_loss(values, td_targets.T)
        self.brain.actor_loss = (-log_probs * advantages.squeeze() - self.brain.beta * entropies).mean()

        return self.brain.critic_loss.item(), self.brain.critic_loss.item()

    def optimize(self):
        self.brain.optimize()

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# print(device)

# a2c = A2C_Feat(state_dim=10, lr_actor=1e-2, lr_critic=1e-2, beta=0.1, gamma=0.98, device=device)
# snake_manager = SnakeEnvParallel(a2c, 1000, dim=5, headless=True, device=device)

# for i in range(1000):
#     average_score, average_game_length, win_percent, max_score, runtime = snake_manager.train_envs()
#     if i%100 == 5:
#         snake_manager.reset_envs(display_example=True)
#     else:
#         snake_manager.reset_envs(display_example=False)

#     print(f"{i}: Avg Score: {average_score:0.3f},\tAvg Game Len: {average_game_length:0.3f},\truntime: {runtime:.6f},\tMax Score: {max_score},\tWin %: {win_percent}")
