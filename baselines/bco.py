import gymnasium as gym
import seals
import sys
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

env_name = "HalfCheetah-v4"  
algo_name = "SAC" 

trajectory_path = f"/home/bavishk/caifo/expert_bc_data/{env_name}_{algo_name}_bcStyle.npy"
expert_trajectory = np.load(trajectory_path, allow_pickle=True)

env = gym.make(env_name)


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
    
    def forward(self, x):
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

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM =  env.action_space.shape[0]
policy = MLP(STATE_DIM, ACTION_DIM)
dummy_policy = SAC("MlpPolicy", env, verbose=1)

#############################################################################

#Rollout with a random policy and then learn the inverse dynamics model

def collect_trajectories(env, policy, n_episodes=100):
    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    #env.seed(SEED)
    MAX_STEPS = 1000
    expert_data = []
    expert_actions = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        step_count = 0
        while not done and step_count<MAX_STEPS:
            action, _states = dummy_policy.predict(state, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            if step_count==0:
                print(f"The state is {state}, the action is {action}, and enxt obs is {next_obs}")
            x = np.concatenate((state[0], next_obs[0])) # x = (s, s')
            expert_data.append(x)
            expert_actions.append(action)
            step_count +=1
            state = next_obs
            
            if done or step_count>=MAX_STEPS:
                break
    return expert_data, expert_actions

env = gym.make(env_name)
rollout, rollout_actions = collect_trajectories(env, policy, n_episodes=1)
print(f"The rollout 1 is {rollout[0]} and 2 is {rollout_actions[0]}" )
idm = InverseDynamics(STATE_DIM, ACTION_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(idm.parameters(), lr=0.001)
rollout = torch.tensor(rollout, dtype = torch.float)
action = torch.tensor(rollout_actions, dtype=torch.float, requires_grad=True).squeeze()#.view(-1, 1)
print(rollout.shape)
print(action.shape)


# num_epochs = 100
# for epoch in range(num_epochs):
#     predicted_actions = idm(rollout)
#     loss = criterion(action, predicted_actions)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
        
# #############################################################################

# state = torch.tensor(expert_trajectory[:, 0], dtype=torch.float)
# inner_shape = expert_trajectory[0].shape[0] * expert_trajectory[0].shape[1]
# expert_trajectory = expert_trajectory.reshape(-1, inner_shape)
# expert_trajectory = torch.tensor(expert_trajectory, dtype = torch.float)
# actions_expert_predicted = idm(expert_trajectory)

# criterion_bc = nn.MSELoss()
# optimizer_bc = optim.Adam(policy.parameters(), lr=0.001)

# num_epochs = 40
# for epoch in range(num_epochs):
#     outputs = policy(state)
#     losses = criterion_bc(outputs, actions_expert_predicted)
#     optimizer_bc.zero_grad()
#     losses.backward(retain_graph=True)
#     optimizer_bc.step()
#     print("POLICY LEARNING")
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}')
    
    
# env = gym.make(env_name)
# NUM_TRAJECTORIES = 10
# MAX_STEPS = 1000  # Maximum number of steps per trajectory
# trajectories = []
# trajectory_rewards = []

# for _ in range(NUM_TRAJECTORIES):
#     trajectory = []
#     rewards = 0  # Variable to accumulate rewards for the trajectory
#     state = env.reset()
#     state = state[0]
#     done = False
#     step_count = 0

#     while not done and step_count < MAX_STEPS:
#         state_tensor = torch.tensor(state, dtype=torch.float)
#         action = policy(state_tensor).detach().numpy()
#         # action = action.detach().item()
#         # discrete_action = 0 if action < 0.5 else 1
#         next_state, reward, done, _ ,_= env.step(action)
#         # trajectory.append((state, discrete_action, reward, next_state, done))
#         rewards += reward  
#         state = next_state
#         step_count += 1

#     # trajectories.append(trajectory)
#     trajectory_rewards.append(rewards)

# # Calculate the mean reward across all trajectories
# mean_reward = np.mean(trajectory_rewards)
# std_reward = np.std(trajectory_rewards)
# print(f"Mean reward over {NUM_TRAJECTORIES} trajectories: {mean_reward} +- {std_reward}")