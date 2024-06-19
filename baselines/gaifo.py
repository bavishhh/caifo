import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
import time
import torch.nn.functional as F
import os
from functools import partial
# from info_nce import InfoNCE

SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)

hparams = {
    "env_name": 'CartPole-v0',   
    
    # Expert
    "total_expert_timesteps": 100000, # no. of timesteps to train the expert
    "num_expert_trajectories": 50, # no. of trajectories to collect from the trained expert
    "expert_algo": "PPO",
    
    # Encoder
    "latent_dim": 64,
    "hidden_dim": 256,
    "encoder_epochs": 100, # no. of epochs to train the encoder inside each PCIL epoch
    "encoder_lr": 0.00001, # learning rate
    
    # InfoNCE Loss
    "batch_size": 128, # number of negative samples
    "expert_data_ratio": 1, # number of positive samples as fraction of negative samples
    
    # Agent
    "epochs": 10, # no. of epochs for the PCIL algorithm
    "learning_steps": 500, # no. of agent updates (using PPO) in each epoch
    "num_agent_trajectories": 10,
    "agent_algo": "PPO",
    "entropy_coeff": 0.01, # entropy coefficient for PPO
}

# optimal parameters can be found at https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
algos = {
    "PPO": partial(PPO, policy="MlpPolicy",
                    n_steps = 1024,
                    batch_size = 64,
                    n_epochs = 4,
                    ent_coef = 0.01,
                    learning_rate = 1e-3,
                    clip_range = 0.2,
                    gae_lambda = 0.98,
                    gamma=0.999,
                    verbose=0, 
                    seed=SEED),
    "DDPG": partial(DDPG, policy="MlpPolicy", verbose=0, seed=SEED),
    "SAC": partial(SAC, policy="MlpPolicy", learning_rate=1e-3, verbose=0, seed=SEED)
}

class AILEnv(gym.Wrapper):
    def __init__(self, env, encoder, expert_buffer):
        super(AILEnv, self).__init__(env)
        self.encoder = encoder
        self.expert_buffer = expert_buffer
        self.curr_obs = None
        self.step_counter = 0
    
    def step(self, action):
        
        next_obs, _, done, truncated, info = self.env.step(action)
        
        assert self.curr_obs is not None
        x_agent = np.concatenate((self.curr_obs[0], next_obs))
        
        # randomly sample a transition from agent buffer.
        x_expert = random.choice(self.expert_buffer)
        
        x_expert = torch.FloatTensor(x_expert).to(device)
        x_agent = torch.FloatTensor(x_agent).to(device)
        
        expert_rep = self.encoder(x_expert)
        agent_rep = self.encoder(x_agent)
        agent_labels = torch.zeros_like(agent_rep)  
        #Reward for the policy
        reward = F.binary_cross_entropy(agent_rep, agent_labels)
        # reward =  torch.cosine_similarity(expert_rep, agent_rep, dim=-1).item()
        
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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
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


def pcil(env, expert_data):
    
    obs = env.reset(seed=SEED)
    
    state_dim = obs[0].shape[0]
    
    encoder = Encoder(state_dim, hparams["hidden_dim"], hparams["latent_dim"])
    encoder.to(device)
    
    criterion = torch.nn.BCELoss()
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=hparams["encoder_lr"])

    agent_buffer = deque(maxlen=10000)
    
    pcil_env = AILEnv(env, encoder, expert_data)
    imitation_agent = algos[hparams["agent_algo"]](env=pcil_env, device=device)
    
    encoder_update_counter = 0
    
    for epoch in range(hparams["epochs"]):
        
        # Collect agent trajectories
        trajectories = collect_trajectories(env, imitation_agent, n_episodes=hparams["num_agent_trajectories"])
        agent_buffer.extend(trajectories)
        
        # Train encoder
        for _ in range(hparams["encoder_epochs"]): 
            
            anchors = random.sample(expert_data,  int(hparams["expert_data_ratio"]*hparams["batch_size"]))
            positives = random.sample(expert_data, int(hparams["expert_data_ratio"]*hparams["batch_size"]))
            negatives = random.sample(agent_buffer, hparams["batch_size"])
            
            anchors = torch.stack(list(map(torch.FloatTensor, anchors))).to(device)
            positives = torch.stack(list(map(torch.FloatTensor, positives))).to(device)
            negatives = torch.stack(list(map(torch.FloatTensor, negatives))).to(device)

            encoder_optimizer.zero_grad()
            
            anchor_reps = encoder(anchors)
            positive_reps = encoder(positives)
            negative_reps = encoder(negatives)
            positive_labels = torch.ones_like(positive_reps)
            negative_labels = torch.zeros_like(negative_reps)  

            positive_loss = F.binary_cross_entropy(positive_reps, positive_labels)
            negative_loss = F.binary_cross_entropy(negative_reps, negative_labels)
            loss = positive_loss + negative_loss
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
    name=f"seed_{SEED}_InfoNCE_{str(time.time())}",
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