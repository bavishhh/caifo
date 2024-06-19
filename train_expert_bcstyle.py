import gymnasium as gym
import seals
import sys
import torch
import numpy as np
import random
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import time
from functools import partial

def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    return _init

def make_env_Multi(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env = Monitor(env)
        return env
    return _f

algos = {
    "PPO": PPO,
    "DDPG": DDPG,
    "SAC": SAC
}
nproc = 8
def train_expert(env_name, training_algo, total_expert_timesteps):
    
    env = SubprocVecEnv([make_env_Multi(env_name, seed) for seed in range(nproc)], start_method = 'fork')
    # env.seed(SEED)
    # expert = algos[training_algo]("MlpPolicy", env, verbose=0, tensorboard_log=f"runsBC/{run.id}", seed=SEED)
    expert_policy_path1 = '/home/bavishk/caifo/experts/expertBC_Ant-v4_SAC.policy'
    expert = partial(SAC, policy="MlpPolicy", verbose=0, ent_coef=0.01, seed=0)(env=env).load(expert_policy_path1, env)
    expert.learn(
        total_timesteps=total_expert_timesteps, 
        progress_bar=True, 
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"modelsBC/{run.id}",
            verbose=2,
        )
    )
    
    mean_reward, _ = evaluate_policy(expert, env, n_eval_episodes=10)
    print(f"Expert's mean reward: {mean_reward}")
    return expert

def collect_trajectories(env, policy, n_episodes=10):

    env = DummyVecEnv([make_env(env.unwrapped.spec.id)])
    expert_data = []
    expert_action = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done : 
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            trajectory = (obs, action, next_obs)
            # expert_trajectories.append(trajectory)
            x = [obs[0], next_obs[0]]
            expert_data.append(x)
            expert_action.append(action[0])
            
            obs = next_obs
            break
            
            if done:
                break
    return expert_data, expert_action


SEED = 0
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# np.random.seed(SEED)
# random.seed(SEED)

env_name = "Ant-v4"
env = gym.make(env_name)

training_algo = "SAC"

expert_policy_path = f"experts/expertBC_{env_name}_{training_algo}.policy"

run = wandb.init(
    project="imitation-learning",
    group=f"Experts",
    name=f"{env_name}_{training_algo}_seedBC_{SEED}_{str(time.time())}",
    sync_tensorboard=True,
    monitor_gym=True,
)

expert = train_expert(env_name, training_algo=training_algo, total_expert_timesteps=1_000_000)
expert.save(expert_policy_path)

#expert_policy_path1 = '/home/bavishk/caifo/experts/expertBC_Ant-v4_SAC.policy'
# expert = partial(SAC, policy="MlpPolicy", verbose=0, ent_coef=0.01, seed=0)(env=env).load(expert_policy_path1, env)
expert_data, expert_action = collect_trajectories(env, expert, 1000)
np.save(f"expert_bc_data/{env_name}_{training_algo}_bcStyle.npy", np.array(expert_data))
np.save(f"expert_bc_data/{env_name}_{training_algo}_bcStyleAction.npy", np.array(expert_action))
# a = np.load('/home/bhavishk/calfo/expert_bc_data/CartPole-v0_PPO_bcStyleAction.npy')
mean_reward, _ = evaluate_policy(expert, env, n_eval_episodes=10)
print(f"Expert's mean reward: {mean_reward}")
