import gymnasium as gym
import seals
import sys
# import gym
# sys.modules["gym"] = gymnasium
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
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env_name = "CartPole-v0"  
algo_name = "PPO" 

trajectory_path = f"/home/bhavishk/calfo/expert_bc_data/{env_name}_{algo_name}_bcStyle.npy"
expert_trajectory = np.load(trajectory_path, allow_pickle=True)
action = np.load(f'/home/bhavishk/calfo/expert_bc_data/{env_name}_{algo_name}_bcStyleAction.npy', allow_pickle=True)

# Define your MLP policy
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

# Initialize the policy
policy = MLP(4, 1)

# Convert numpy arrays to PyTorch tensors
state = torch.tensor(expert_trajectory[:, 0], dtype=torch.float)
action = torch.tensor(action, dtype=torch.float).view(-1, 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    outputs = policy(state)
    loss = criterion(outputs, action)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
print("#####################################POLICY TRAINED ")
import pickle

with open(f'baselines/policies/{env_name}_{algo_name}.pkl', 'wb') as f:
    pickle.dump(policy, f)
    
print("##########################MODEL SAVED")

# Define a function to convert the policy into a function compatible with Stable Baselines
def mlp_policy(state):
    state_tensor = torch.tensor(state, dtype=torch.float)
    action = policy(state_tensor)
    return action.detach().numpy()


env = gym.make(env_name)
NUM_TRAJECTORIES = 20
MAX_STEPS = 200  # Maximum number of steps per trajectory
trajectories = []
trajectory_rewards = []

for _ in range(NUM_TRAJECTORIES):
    trajectory = []
    rewards = 0  # Variable to accumulate rewards for the trajectory
    state = env.reset()
    state = state[0]
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        state_tensor = torch.tensor(state, dtype=torch.float)
        action = policy(state_tensor)
        action = action.detach().item()
        discrete_action = 0 if action < 0.5 else 1
        next_state, reward, done, _ ,_= env.step(discrete_action)
        trajectory.append((state, discrete_action, reward, next_state, done))
        rewards += reward  # Accumulate rewards
        state = next_state
        step_count += 1

    trajectories.append(trajectory)
    trajectory_rewards.append(rewards)

# Calculate the mean reward across all trajectories
mean_reward = np.mean(trajectory_rewards)
std_reward = np.std(trajectory_rewards)
print(f"Mean reward over {NUM_TRAJECTORIES} trajectories: {mean_reward} +- {std_reward}")


