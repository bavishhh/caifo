import torch
import random
import gymnasium as gym
import numpy as np


class CALFOEnv(gym.Wrapper):
    def __init__(self, env, encoder, expert_buffer, device="cpu"):
        super(CALFOEnv, self).__init__(env)
        self.encoder = encoder
        self.expert_buffer = expert_buffer
        self.curr_obs = None
        self.step_counter = 0
        self.device = device
    
    def step(self, action):
        
        next_obs, _, done, truncated, info = self.env.step(action)
        
        assert self.curr_obs is not None
        x_agent = np.concatenate((self.curr_obs[0], next_obs))
        
        # randomly sample a transition from agent buffer.
        x_expert = random.choice(self.expert_buffer)
        
        x_expert = torch.FloatTensor(x_expert).to(self.device)
        x_agent = torch.FloatTensor(x_agent).to(self.device)
        
        expert_rep = self.encoder(x_expert)
        agent_rep = self.encoder(x_agent)
        
        reward =  torch.cosine_similarity(expert_rep, agent_rep, dim=-1).item()
        # reward = -torch.sqrt(torch.sum(torch.pow(expert_rep - agent_rep, 2), dim=0))  
        
        self.step_counter += 1
        
        return next_obs, reward, done, truncated, info
    
    def update_encoder(self, encoder):
        self.encoder = encoder
        
    def reset(self, **kwargs):
        self.curr_obs = self.env.reset(**kwargs)
        return self.curr_obs
