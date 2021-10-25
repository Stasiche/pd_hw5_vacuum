import torch

from DQN import DQN
import random
from wrapper import Wrapper
import yaml
from tqdm import trange
from utils.wandb_stuff import wandb_init, wandb_log
from utils.evaluation import evaluate_policy, generate_gif
import wandb
import os
from os.path import join

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

wandb_init(config)
exp_dir_name = join('experiments', wandb.run.name)
os.makedirs(exp_dir_name, exist_ok=True)

env = Wrapper(**config['env'])
dqn = DQN(env.action_space.n, config['train']['buffer_size'], config['train']['batch_size'], config['DQN'])
eps = 0.1
state = env.reset()

for _ in trange(config['train']['buffer_size']):
    action = env.action_space.sample()

    next_state, reward, done, _, _ = env.step(action)
    dqn.consume_transition((state, action, next_state, reward, done))

    state = next_state if not done else env.reset()

for i in trange(config['train']['transitions']):
    # Epsilon-greedy policy
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        action = dqn.act(state, eps, i)

    next_state, reward, done, _, _ = env.step(action)
    dqn.update((state, action, next_state, reward, done))

    state = next_state if not done else env.reset()

    if not (i+1) % (config['train']['transitions'] // 100):
        reward_mean, reward_std, metric_mean, metric_std = evaluate_policy(config['env'], dqn, 5)
        dqn.save(i, metric_mean, exp_dir_name)
        wandb_log(exp_dir_name, reward_mean, reward_std, metric_mean, metric_std)
        generate_gif(config['env'], dqn)

generate_gif(config["env"], dqn)
