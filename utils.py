import gymnasium as gym
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wandb

def collect_trajectories(env, policy, n_episodes=10, seed=0):
    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    env.seed(seed)
    data = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            
            x = np.concatenate((obs[0], next_obs[0])) # x = (s, s')
            data.append(x)
            
            obs = next_obs
            
            if done:
                break
         
    return data


def plot_reps(encoder, expert_data, agent_data, name="PCA", device="cpu"):

    expert_data = torch.stack(list(map(torch.FloatTensor, expert_data))).to(device)
    agent_data = torch.stack(list(map(torch.FloatTensor, agent_data))).to(device)

    expert_reps = encoder(expert_data)
    agent_reps = encoder(agent_data)
    
    data = torch.concatenate([expert_reps, agent_reps]).cpu()
    # result = umap.UMAP().fit_transform(data)
    result = PCA(n_components=2).fit_transform(data)
    labels = [1]*expert_data.size(0) + [0]*agent_data.size(0)
    
    plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=labels)
    wandb.log({f"{name}": plt})


def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    return _init