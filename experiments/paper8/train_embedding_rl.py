# -*- coding: utf-8 -*-
"""
Paper 8: Train RL policies on EmbeddingSpaceEnv.

Trains MaskablePPO with a simple MLP policy on the embedding-space environment.
Evaluates in both embedding space and (optionally) real environment via translator.

Usage:
    python paper8/train_embedding_rl.py                    # single seed
    python paper8/train_embedding_rl.py --n_seeds 15       # multi-seed
    python paper8/train_embedding_rl.py --method greedy     # greedy baseline
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from embedding_space_env import (
    EmbeddingSpaceEnv, load_ldn_model, load_lulc_decoder,
    SCENARIO_NAMES, CROPLAND_CLASS
)

PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PAPER8_DIR, 'data')
RESULTS_DIR = os.path.join(PAPER8_DIR, 'results')


def make_env(n_regions=50, max_steps=5):
    """Create EmbeddingSpaceEnv with Bishan 2020 data."""
    emb = np.load(os.path.join(DATA_DIR, 'bishan_emb_2020.npy'))
    ctx = np.load(os.path.join(DATA_DIR, 'bishan_context.npy'))
    ldn = load_ldn_model()
    decoder = load_lulc_decoder()
    env = EmbeddingSpaceEnv(emb, ctx, ldn, decoder,
                            n_regions=n_regions, max_steps=max_steps)
    return env


def evaluate_embedding(env, policy_fn, n_episodes=5):
    """Evaluate a policy in embedding space.

    Args:
        env: EmbeddingSpaceEnv
        policy_fn: callable(obs, env) -> action [region_id, scenario_id]
        n_episodes: number of evaluation episodes

    Returns:
        dict with mean metrics
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        lulc_init = info['lulc']
        crop_init = (lulc_init == CROPLAND_CLASS).sum()
        done = False
        total_reward = 0
        actions = []

        while not done:
            action = policy_fn(obs, env)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            actions.append((int(action[0]), int(action[1])))
            done = done or trunc

        crop_final = info['cropland_after']
        results.append({
            'reward': total_reward,
            'cropland_init': int(crop_init),
            'cropland_final': int(crop_final),
            'cropland_change': int(crop_final - crop_init),
            'actions': actions,
        })

    return {
        'mean_reward': np.mean([r['reward'] for r in results]),
        'std_reward': np.std([r['reward'] for r in results]),
        'mean_cropland_change': np.mean([r['cropland_change'] for r in results]),
        'episodes': results,
    }


def random_policy(obs, env):
    """Random action selection."""
    return env.action_space.sample()


def greedy_policy(obs, env):
    """Greedy: pick highest-intensity on least-intervened region."""
    max_intensity_idx = len(env.action_space.nvec) - 1  # highest intensity
    for r in range(env.n_regions):
        if r not in env.intervened:
            return np.array([r, min(4, env.action_space.nvec[1] - 1)])
    return env.action_space.sample()


def train_ppo(seed, n_regions=50, max_steps=5, timesteps=50_000, out_dir=None):
    """Train PPO on EmbeddingSpaceEnv."""
    if out_dir is None:
        out_dir = os.path.join(RESULTS_DIR, 'ppo')
    os.makedirs(out_dir, exist_ok=True)

    eval_path = os.path.join(out_dir, f'eval_seed{seed}.json')
    if os.path.exists(eval_path):
        print(f'  [seed {seed}] Already done, skipping')
        with open(eval_path) as f:
            return json.load(f)

    env = make_env(n_regions=n_regions, max_steps=max_steps)
    env = Monitor(env)

    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=max_steps * 10,  # collect 10 episodes per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    # Save model
    model_path = os.path.join(out_dir, f'model_seed{seed}.zip')
    model.save(model_path)

    # Evaluate
    eval_env = make_env(n_regions=n_regions, max_steps=max_steps)

    def ppo_policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    metrics = evaluate_embedding(eval_env, ppo_policy, n_episodes=5)

    result = {
        'seed': seed,
        'method': 'ppo',
        'training_time_s': train_time,
        'timesteps': timesteps,
        'n_regions': n_regions,
        'max_steps': max_steps,
        **{k: v for k, v in metrics.items() if k != 'episodes'},
    }

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  [seed {seed}] reward={result["mean_reward"]:.2f}, '
          f'crop_change={result["mean_cropland_change"]:.1f}, '
          f'time={train_time:.0f}s')
    return result


def run_baseline(method, n_regions=50, max_steps=5, n_seeds=5):
    """Run random or greedy baseline."""
    out_dir = os.path.join(RESULTS_DIR, method)
    os.makedirs(out_dir, exist_ok=True)

    policy_fn = random_policy if method == 'random' else greedy_policy
    all_results = []

    for seed in range(n_seeds):
        eval_path = os.path.join(out_dir, f'eval_seed{seed}.json')
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                all_results.append(json.load(f))
            print(f'  [{method} seed {seed}] Already done')
            continue

        env = make_env(n_regions=n_regions, max_steps=max_steps)
        np.random.seed(seed)
        metrics = evaluate_embedding(env, policy_fn, n_episodes=5)

        result = {
            'seed': seed,
            'method': method,
            'n_regions': n_regions,
            'max_steps': max_steps,
            **{k: v for k, v in metrics.items() if k != 'episodes'},
        }
        with open(eval_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'  [{method} seed {seed}] reward={result["mean_reward"]:.2f}, '
              f'crop_change={result["mean_cropland_change"]:.1f}')
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ppo', 'random', 'greedy', 'all'],
                        default='all')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_regions', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--timesteps', type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    if args.method in ('random', 'all'):
        print('=== Random Baseline ===')
        run_baseline('random', args.n_regions, args.max_steps, args.n_seeds)

    if args.method in ('greedy', 'all'):
        print('=== Greedy Baseline ===')
        run_baseline('greedy', args.n_regions, args.max_steps, args.n_seeds)

    if args.method in ('ppo', 'all'):
        print('=== PPO Training ===')
        for seed in range(args.n_seeds):
            train_ppo(seed, args.n_regions, args.max_steps, args.timesteps)

    # Summary
    print(f'\n{"="*50}')
    print(f'Total time: {(time.time()-t0)/60:.1f} min')
    for method in ['random', 'greedy', 'ppo']:
        d = os.path.join(RESULTS_DIR, method)
        if not os.path.exists(d):
            continue
        evals = []
        for f in sorted(os.listdir(d)):
            if f.startswith('eval_') and f.endswith('.json'):
                with open(os.path.join(d, f)) as fh:
                    evals.append(json.load(fh))
        if evals:
            rewards = [e['mean_reward'] for e in evals]
            crops = [e['mean_cropland_change'] for e in evals]
            print(f'{method:>8}: reward={np.mean(rewards):+.2f}+-{np.std(rewards):.2f}, '
                  f'crop_change={np.mean(crops):+.1f}+-{np.std(crops):.1f} '
                  f'({len(evals)} seeds)')


if __name__ == '__main__':
    main()
