# -*- coding: utf-8 -*-
"""
Paper 8: Ablation experiments on county-level data.

Ablation 1: Region granularity K = {25, 50, 100, 200}
Ablation 2: Planning horizon T = {3, 5, 10}
Ablation 3: Reward weights (cropland_weight, coherence_weight, forest_penalty)

Usage:
    python paper8/ablation_experiments.py                    # all
    python paper8/ablation_experiments.py --ablation regions  # K only
    python paper8/ablation_experiments.py --ablation horizon  # T only
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from embedding_space_env import (
    EmbeddingSpaceEnv, load_ldn_model, load_lulc_decoder,
    INTENSITY_LEVELS, CROPLAND_CLASS
)

PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PAPER8_DIR, 'data')
RESULTS_DIR = os.path.join(PAPER8_DIR, 'results', 'ablation')

N_SEEDS = 5
TIMESTEPS = 50_000


def evaluate(env, policy_fn, n_episodes=5):
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        crop_init = (info['lulc'] == CROPLAND_CLASS).sum()
        done, total_reward = False, 0
        while not done:
            action = policy_fn(obs, env)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            done = done or trunc
        results.append({
            'reward': total_reward,
            'cropland_change': int(info['cropland_after'] - crop_init),
        })
    return {
        'mean_reward': float(np.mean([r['reward'] for r in results])),
        'mean_cropland_change': float(np.mean([r['cropland_change'] for r in results])),
    }


def train_and_eval(seed, env_factory, out_dir, label, timesteps=TIMESTEPS):
    os.makedirs(out_dir, exist_ok=True)
    eval_path = os.path.join(out_dir, f'{label}_eval_seed{seed}.json')
    if os.path.exists(eval_path):
        print(f'    [{label} seed {seed}] exists, skip')
        with open(eval_path) as f:
            return json.load(f)

    env = Monitor(env_factory())
    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=env.env.max_steps * 10,
        batch_size=50,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, seed=seed, verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    eval_env = env_factory()
    def ppo_fn(obs, e):
        a, _ = model.predict(obs, deterministic=True)
        return a

    metrics = evaluate(eval_env, ppo_fn)
    result = {'seed': seed, 'label': label, 'training_time_s': train_time, **metrics}

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'    [{label} seed {seed}] crop={result["mean_cropland_change"]:+.0f}, '
          f'reward={result["mean_reward"]:+.1f}, {train_time:.0f}s')
    return result


def ablation_regions():
    """Ablation 1: Region granularity K."""
    print('\n' + '='*60)
    print('ABLATION 1: Region granularity K')
    print('='*60)

    ldn, decoder = load_ldn_model(), load_lulc_decoder()
    emb = np.load(os.path.join(DATA_DIR, 'bishan_emb_2020.npy'))
    ctx = np.load(os.path.join(DATA_DIR, 'bishan_context.npy'))
    out_dir = os.path.join(RESULTS_DIR, 'regions')

    for K in [25, 50, 100, 200]:
        label = f'K{K}'
        print(f'\n  K={K}:')

        def factory(k=K):
            return EmbeddingSpaceEnv(emb, ctx, ldn, decoder,
                                     n_regions=k, max_steps=5)

        for seed in range(N_SEEDS):
            train_and_eval(seed, factory, out_dir, label)

    # Summary
    print(f'\n  === Region Ablation Summary ===')
    for K in [25, 50, 100, 200]:
        label = f'K{K}'
        crops = []
        for seed in range(N_SEEDS):
            f = os.path.join(out_dir, f'{label}_eval_seed{seed}.json')
            if os.path.exists(f):
                crops.append(json.load(open(f))['mean_cropland_change'])
        if crops:
            print(f'  K={K:>3}: crop={np.mean(crops):+.0f}+-{np.std(crops):.0f}')


def ablation_horizon():
    """Ablation 2: Planning horizon T."""
    print('\n' + '='*60)
    print('ABLATION 2: Planning horizon T')
    print('='*60)

    ldn, decoder = load_ldn_model(), load_lulc_decoder()
    emb = np.load(os.path.join(DATA_DIR, 'bishan_emb_2020.npy'))
    ctx = np.load(os.path.join(DATA_DIR, 'bishan_context.npy'))
    out_dir = os.path.join(RESULTS_DIR, 'horizon')

    for T in [3, 5, 10]:
        label = f'T{T}'
        print(f'\n  T={T}:')

        def factory(t=T):
            return EmbeddingSpaceEnv(emb, ctx, ldn, decoder,
                                     n_regions=50, max_steps=t)

        for seed in range(N_SEEDS):
            train_and_eval(seed, factory, out_dir, label)

    # Summary
    print(f'\n  === Horizon Ablation Summary ===')
    for T in [3, 5, 10]:
        label = f'T{T}'
        crops, rewards = [], []
        for seed in range(N_SEEDS):
            f = os.path.join(out_dir, f'{label}_eval_seed{seed}.json')
            if os.path.exists(f):
                d = json.load(open(f))
                crops.append(d['mean_cropland_change'])
                rewards.append(d['mean_reward'])
        if crops:
            print(f'  T={T:>2}: crop={np.mean(crops):+.0f}+-{np.std(crops):.0f}, '
                  f'reward={np.mean(rewards):+.1f}+-{np.std(rewards):.1f}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', choices=['regions', 'horizon', 'all'], default='all')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    if args.ablation in ('regions', 'all'):
        ablation_regions()

    if args.ablation in ('horizon', 'all'):
        ablation_horizon()

    print(f'\n\nTotal elapsed: {(time.time()-t0)/3600:.1f} hours')


if __name__ == '__main__':
    main()
