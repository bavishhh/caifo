import gymnasium as gym
import seals
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
import time
import os
from functools import partial

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)

hparams = {
    "env_name": 'CartPole-v1',   
    
    # Expert
    "total_expert_timesteps": 100000, # no. of timesteps to train the expert
    "num_expert_trajectories": 50, # no. of trajectories to collect from the trained expert
    "expert_algo": "PPO",
    
    # Encoder
    "latent_dim": 64,
    "hidden_dim": 256,
    "encoder_epochs": 10, # no. of epochs to train the encoder inside each PCIL epoch
    "encoder_lr": 5e-4, # learning rate
    
    # InfoNCE Loss
    "batch_size": 128, # number of negative samples
    "expert_data_ratio": 0.5, # number of positive samples as fraction of negative samples
    
    # Agent
    "epochs": 20, # no. of epochs for the PCIL algorithm
    "learning_steps": 1000, # no. of agent updates (using PPO) in each epoch
    "num_agent_trajectories": 10,
    "agent_algo": "PPO",
    "entropy_coeff": 0.001, # entropy coefficient for PPO
}

algos = {
    "PPO": partial(PPO, policy="MlpPolicy", verbose=0, ent_coef=hparams["entropy_coeff"], seed=SEED),
    "DDPG": partial(DDPG, policy="MlpPolicy", verbose=0, seed=SEED)
}

class PCILEnv(gym.Wrapper):
    def __init__(self, env, encoder, expert_buffer):
        super(PCILEnv, self).__init__(env)
        self.encoder = encoder
        self.expert_buffer = expert_buffer
        self.curr_obs = None
        self.step_counter = 0
    
    def step(self, action):
        
        next_obs, _, done, truncated, info = self.env.step(action)
        
        assert self.curr_obs is not None
        x_agent = np.concatenate((self.curr_obs[0], next_obs))
        x_expert = random.choice(self.expert_buffer)
        
        x_expert = torch.FloatTensor(x_expert).to(device)
        x_agent = torch.FloatTensor(x_agent).to(device)
        
        reward = pcil_reward(self.encoder, x_expert, x_agent)
        
        self.step_counter += 1
        
        return next_obs, reward, done, truncated, info
    
    def update_encoder(self, encoder):
        self.encoder = encoder
        
    def reset(self, **kwargs):
        self.curr_obs = self.env.reset()
        return self.curr_obs


class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    return _init


def collect_trajectories(env, policy, n_episodes=10):
    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    env.seed(SEED)
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


def infoNCE_loss(anchor, positive, negatives, temperature=0.07):
    if anchor.dim() == 1:
        anchor = anchor.unsqueeze(0)
    if positive.dim() == 1:
        positive = positive.unsqueeze(0)
    if negatives.dim() == 2:
        negatives = negatives.unsqueeze(1)

    # Compute similarities
    pos_sim = torch.exp(torch.cosine_similarity(anchor, positive, dim=1) / temperature)
    
    # Reshape for broadcasting: (batch_size, 1, latent_dim) and (1, num_negatives, latent_dim)
    anchor_expanded = anchor.unsqueeze(1)
    negatives_expanded = negatives.transpose(0, 1)
    
    # Compute pairwise cosine similarities: (batch_size, num_negatives)
    neg_sims = torch.cosine_similarity(anchor_expanded, negatives_expanded, dim=2)
    neg_sims = torch.exp(neg_sims / temperature)
    
    # Sum over negative samples
    denominator = pos_sim + neg_sims.sum(dim=1)
    
    # Compute loss
    loss = -torch.log(pos_sim / denominator)
    return loss.mean()


def pcil_reward(encoder, expert_state, agent_state):
    expert_rep = encoder(expert_state)
    agent_rep = encoder(agent_state)
    return torch.cosine_similarity(expert_rep, agent_rep, dim=-1).item()


def pcil(env, expert_data):
    
    obs = env.reset(seed=SEED)
    
    state_dim = obs[0].shape[0]
    
    encoder = Encoder(state_dim, hparams["hidden_dim"], hparams["latent_dim"])
    encoder.to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=hparams["encoder_lr"])

    agent_buffer = deque(maxlen=10000)
    
    pcil_env = PCILEnv(env, encoder, expert_data)
    imitation_agent = algos[hparams["agent_algo"]](env=pcil_env, device=device)
    
    encoder_update_counter = 0
    
    for epoch in range(hparams["epochs"]):
        
        # Collect agent trajectories
        agent_buffer.extend(collect_trajectories(env, imitation_agent, n_episodes=hparams["num_agent_trajectories"]))
        
        # Train encoder
        for _ in range(hparams["encoder_epochs"]): 
            
            anchors = random.sample(expert_data, 1)
            positives = random.sample(expert_data, int(hparams["expert_data_ratio"]*hparams["batch_size"]))
            negatives = random.sample(agent_buffer, hparams["batch_size"])
            
            anchors = torch.stack(list(map(torch.FloatTensor, anchors))).to(device)
            positives = torch.stack(list(map(torch.FloatTensor, positives))).to(device)
            negatives = torch.stack(list(map(torch.FloatTensor, negatives))).to(device)

            anchor_reps = encoder(anchors)
            positive_reps = encoder(positives)
            negatives_reps = encoder(negatives)
            
            loss = infoNCE_loss(anchor_reps, positive_reps, negatives_reps)
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            
            encoder_update_counter += 1
            wandb.log({"imitator/encoder_loss": loss.item(), "imitator/enc_update_count": encoder_update_counter})
            
        # Update the env with trained encoder
        pcil_env.update_encoder(encoder)
        
        # Train imitation agent
        imitation_agent.learn(total_timesteps=hparams["learning_steps"])
        
        # Evaluate imitation agent's performance on the original environment
        mean_reward, _ = evaluate_policy(imitation_agent, env, n_eval_episodes=5)
        
        print(f"Epoch {epoch+1}: Imitation Agent's mean reward = {mean_reward}")
        wandb.log({"imitator/mean_reward": mean_reward, "imitator/env_step": pcil_env.step_counter})

    return imitation_agent

# Main

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

env_name = hparams["env_name"]
env = gym.make(env_name)
env = Monitor(env)

run = wandb.init(
    project="imitation-learning",
    group=f"{env_name}-{hparams['agent_algo']}",
    name=f"seed_{SEED}_{str(time.time())}",
    sync_tensorboard=True,
    monitor_gym=True,
    config={
        **hparams,
        "device": device
    }
)

expert_data_path = f"expert_data/{env_name}_{hparams['expert_algo']}.npy"
expert_policy_path = f"experts/expert_{env_name}_{hparams['expert_algo']}.policy"

if os.path.exists(expert_data_path):
    print("Loading expert data")
    expert_data = np.load(expert_data_path).tolist()
    
elif os.path.exists(expert_policy_path):
    print("Loading trained expert policy...")
    expert = algos[hparams["expert_algo"]](env=env).load(expert_policy_path, env)
    
    print("Collecting expert trajectories...")
    expert_data = collect_trajectories(env, expert, n_episodes=hparams["num_expert_trajectories"])
    
else:
    raise "Expert data or policy does not exist"
    
print("Num expert transitions:", len(expert_data))

imitation_agent = pcil(env, expert_data)

mean_reward, _ = evaluate_policy(imitation_agent, env, n_eval_episodes=10)
print(f"Final Imitation Agent's mean reward: {mean_reward}")