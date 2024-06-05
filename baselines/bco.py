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

TRAJECTORY_PATH = " "

trajectory_path = TRAJECTORY_PATH
expert_trajectory = np.load(trajectory_path, allow_pickle=True)

STATE_DIM = expert_trajectory[0][0][0].shape[0]
ACTION_DIM = expert_trajectory[0][1][0].shape[0]

#########################################################################

#Policy, IDM Definition


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class InverseDynamics(nn.Module):
    def __init__(self, state_dim, action_space):
        super(InverseDynamics, self).__init__()
        self.fc1 = nn.Linear(state_dim*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    return _init

##########################################################################

#Initialize variables for idm training 

policy = MLP(STATE_DIM, ACTION_DIM)

#############################################################################

#Rollout with a random policy and then learn the inverse dynamics model

def collect_trajectories(env, policy, n_episodes=10):
    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    #env.seed(SEED)
    expert_data = []
    for _ in range(n_episodes):
        obs = env.reset(seed = SEED)
        done = False
        while not done:
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            x.append([obs[0]])
            x.append([action])
            x.append([next_obs[0]])
            x = np.array(x)
            expert_data.append(x)
            
            obs = next_obs
            
            if done:
                break
    return expert_data

rollout = collect_trajectories(env, policy, n_episodes=10)
idm = InverseDynamics(STATE_DIM, ACTION_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(idm.parameters(), lr=0.00001) 
state = torch.stack([torch.tensor(arr[0]) for arr in rollout[:, 0]])
action = torch.stack([torch.tensor(arr[0]) for arr in rollout[:, 1]])
next_state = torch.stack([torch.tensor(arr[0]) for arr in rollout[:, 2]])

num_epochs = 10000
for epoch in range(num_epochs):
    predicted_actions = idm(state, next_state)
    loss = criterion(predicted_actions, action)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
        
#############################################################################

#Predict actions for each state, next state pair and proceed with BC

state_expert = torch.stack([torch.tensor(arr[0]) for arr in expert_trajectory[:, 0]])
next_state_expert = torch.stack([torch.tensor(arr[0]) for arr in expert_trajectory[:, 2]])
actions_expert_predicted = idm(state_expert, next_state_expert)

criterion_bc = nn.MSELoss()
optimizer_bc = optim.Adam(policy.parameters(), lr=0.00001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = policy(state_expert)
    loss = criterion_bc(outputs, actions_expert_predicted)
    optimizer_bc.zero_grad()
    loss.backward()
    optimizer.step()
    # wandb.log({'loss': loss.item()})

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')