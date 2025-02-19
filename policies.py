import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR

class A2C_10_Features(nn.Module):
    """
    Advantage Actor-Critic (A2C) neural network with feature extraction.
    
    Parameters:
        state_dim (int): Dimension of input state
        action_dim (int): Number of possible actions
        gamma (float): Discount factor for future rewards
        lr_actor (float): Learning rate for actor network
        lr_critic (float): Learning rate for critic network
        beta (float): Entropy coefficient for exploration
    """
    def __init__(self, state_dim, action_dim=4, gamma=0.99, beta = 0.01, device=torch.device('cpu'),
                 lr_actor=1e-2, actor_step_size = 10, actor_gamma = 0.1, min_lr_actor = 1e-4,
                 lr_critic=1e-2, critic_step_size = 10, critic_gamma = 0.1, min_lr_critic = 1e-4):
        
        super(A2C_10_Features, self).__init__()

        self.device = device
        self.gamma = gamma
        self.beta = beta

        # Define feature extraction net
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        ).to(self.device)

        # Actor network
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=actor_step_size, gamma=actor_gamma)
        self.min_lr_actor = min_lr_actor
        self.actor_loss = 0

        # Critic network
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=critic_step_size, gamma=critic_gamma)
        self.min_lr_critic = min_lr_critic
        self.critic_loss = 0

    def optimize(self):
        """
        Performs optimization step for both actor and critic networks using accumulated losses.
        """
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        total_loss = self.critic_loss + self.actor_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        self.actor_loss = 0
        self.critic_loss = 0 

    def get_action(self, state):
        """
        Selects action from current policy for given state.
        
        Parameters:
            state (tensor or ndarray): Current game state
        
        Returns:
            tuple: (action, log_probability, entropy, value_estimate)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)

        # print(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        # print(dist.probs)
        actions = dist.sample()

        return actions.cpu().numpy(), dist.log_prob(actions), dist.entropy(), value
    
    def get_value(self, state, nograd=False):
        """
        Computes value estimate for given state.
        
        Parameters:
            state (tensor or ndarray): Current game state
            nograd (bool): Whether to compute gradients
        
        Returns:
            tensor: Value estimate for the state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        if nograd:
            with torch.no_grad():
                features = self.features(state)
                value = self.critic(features)
        else:
            features = self.features(state)
            value = self.critic(features)
        
        return value