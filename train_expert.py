import gymnasium as gym
import seals
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

def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    return _init

algos = {
    "PPO": PPO,
    "DDPG": DDPG,
    "SAC": SAC
}

def train_expert(env_name, training_algo, total_expert_timesteps):
    
    env = DummyVecEnv([make_env(env_name)])
    env.seed(SEED)
    expert = algos[training_algo]("MlpPolicy", env, verbose=0, tensorboard_log=f"runs/{run.id}", seed=SEED)
    
    expert.learn(
        total_timesteps=total_expert_timesteps, 
        progress_bar=True, 
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )
    
    mean_reward, _ = evaluate_policy(expert, env, n_eval_episodes=10)
    print(f"Expert's mean reward: {mean_reward}")
    return expert

def collect_trajectories(env, policy, n_episodes=10):
    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    env.seed(SEED)
    expert_data = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            
            x = np.concatenate((obs[0], next_obs[0])) # x = (s, s')
            expert_data.append(x)
            
            obs = next_obs
            
            if done:
                break
    return expert_data


SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
random.seed(SEED)

env_name = "CartPole-v0"
env = gym.make(env_name)

training_algo = "PPO"

expert_policy_path = f"experts/expert_{env_name}_{training_algo}.policy"

run = wandb.init(
    project="imitation-learning",
    group=f"Experts",
    name=f"{env_name}_{training_algo}_seed_{SEED}_{str(time.time())}",
    sync_tensorboard=True,
    monitor_gym=True,
)


expert = train_expert(env_name, training_algo=training_algo, total_expert_timesteps=100_000)
expert.save(expert_policy_path)

expert_data = collect_trajectories(env, expert, 10)
np.save(f"expert_data/{env_name}_{training_algo}.npy", np.array(expert_data))