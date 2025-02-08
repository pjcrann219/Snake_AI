import numpy as np
import torch
import torch.nn as nn


class A2C_CNN(nn.Module):
    def __init__(self, state_dim, action_dim=4, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, beta = 0.01):
        super(A2C_CNN, self).__init__()

        self.gamma = gamma
        self.beta = beta

        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        example_features = self.features(torch.randn(1, 4, state_dim, state_dim))
        feature_map_size = example_features.numel()

        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_map_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_loss = 0

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_map_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = 0

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)

        print(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        # print(dist.probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy().mean(), value
    
    def get_value(self, state, nograd=False):
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
    
# a2c = A2C(16)
# state = torch.randn(1, 3, 16, 16)
# features = a2c.features(state)
# print(features.size())
# action_probs = a2c.actor(features)
# print(action_probs)
# value = a2c.critic(features)
# print(value)

class A2C_FC(nn.Module):
    def __init__(self, state_dim, action_dim=4, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, beta = 0.01):
        super(A2C_FC, self).__init__()

        self.gamma = gamma
        self.beta = beta

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * state_dim**2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_loss = 0

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = 0

    def update_loss(self, entropy, log_prob, value, next_state, reward, done, optimize=True):

        if done:
            next_value = torch.tensor([[0]])
        else:
            done = False
            next_state_torch = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_value = self.get_value(next_state_torch, nograd=True) # Do we need no grad if we use td_target.detach() below?

        # Compute advantage function
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        # Update critic/actor loss
        self.critic_loss += torch.nn.functional.mse_loss(value, td_target.detach())
        self.actor_loss += -log_prob * advantage.detach() - self.beta * entropy
        
        if optimize:
            self.optimize

    def optimize(self):
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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)

        # print(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        # print(dist.probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy().mean(), value
    
    def get_value(self, state, nograd=False):
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
    
# a2c = A2C(16)
# state = torch.randn(1, 3, 16, 16)
# features = a2c.features(state)
# print(features.size())
# action_probs = a2c.actor(features)
# print(action_probs)
# value = a2c.critic(features)
# print(value)


class A2C_ViT(nn.Module):
    def __init__(self, state_dim, action_dim=4, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, beta = 0.01):
        super(A2C_FC, self).__init__()

        self.gamma = gamma
        self.beta = beta

        self.d_model = 3 * state_dim**2
        self.nhead = 8

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * state_dim**2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_loss = 0

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = 0

    def update_loss(self, entropy, log_prob, value, next_state, reward, done, optimize=True):

        if done:
            next_value = torch.tensor([[0]])
        else:
            done = False
            next_state_torch = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_value = self.get_value(next_state_torch, nograd=True) # Do we need no grad if we use td_target.detach() below?

        # Compute advantage function
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        # Update critic/actor loss
        self.critic_loss += torch.nn.functional.mse_loss(value, td_target.detach())
        self.actor_loss += -log_prob * advantage.detach() - self.beta * entropy
        
        if optimize:
            self.optimize

    def optimize(self):
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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)

        # print(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        # print(dist.probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy().mean(), value
    
    def get_value(self, state, nograd=False):
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
    