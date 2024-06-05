import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium
# import seals
import sys
import gym
sys.modules["gym"] = gymnasium
import torch
import numpy as np
import random
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import time

trajectory_path = r'C:\Users\hrida\caifo\expert_data\Pendulum-v1_SAC_bcStyle.npy'
expert_trajectory = np.load(trajectory_path, allow_pickle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
policy = MLP(3, 1)
state = torch.stack([torch.tensor(arr[0]) for arr in expert_trajectory[:, 0]])
action = torch.stack([torch.tensor(arr[0]) for arr in expert_trajectory[:, 1]])

criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = policy(state)
    loss = criterion(outputs, action)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # wandb.log({'loss': loss.item()})

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')