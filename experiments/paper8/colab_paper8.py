# -*- coding: utf-8 -*-
"""
Paper 8 Colab Training Script — All Experiments.

Upload paper8_colab.zip to Google Drive, then run in Colab:

    !pip install gymnasium scikit-learn joblib
    %cd /content
    !cp /content/drive/MyDrive/paper8_colab.zip .
    !unzip -qo paper8_colab.zip
    !python colab_paper8.py                          # all experiments
    !python colab_paper8.py --exp county              # county only
    !python colab_paper8.py --exp village              # village only
    !python colab_paper8.py --exp transfer             # transfer only

Results saved to results/ and synced back to Drive.

Estimated CPU times:
    County (15 seeds × 50K):   ~10 h
    Village (15 seeds × 50K):  ~15 h
    Transfer (15 seeds eval):  ~1 h
    Total: ~26 h (can run overnight on Colab Pro+)
"""

import os
import sys
import json
import time
import shutil
import argparse
import numpy as np

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from embedding_space_env import (
    EmbeddingSpaceEnv, load_ldn_model, load_lulc_decoder,
    INTENSITY_LEVELS, CROPLAND_CLASS
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DRIVE_SYNC = '/content/drive/MyDrive/paper8_results'


def sync_to_drive():
    """Copy results to Google Drive for persistence."""
    if os.path.exists('/content/drive'):
        os.makedirs(DRIVE_SYNC, exist_ok=True)
        for root, dirs, files in os.walk(RESULTS_DIR):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, RESULTS_DIR)
                dst = os.path.join(DRIVE_SYNC, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        print(f'  [Synced to Drive: {DRIVE_SYNC}]')


def make_env(data_dir, emb_file, ctx_file, n_regions, max_steps=5, ldn=None, decoder=None):
    emb = np.load(os.path.join(data_dir, emb_file))
    ctx = np.load(os.path.join(data_dir, ctx_file))
    if ldn is None:
        ldn = load_ldn_model()
    if decoder is None:
        decoder = load_lulc_decoder()
    return EmbeddingSpaceEnv(emb, ctx, ldn, decoder,
                             n_regions=n_regions, max_steps=max_steps)


def evaluate(env, policy_fn, n_episodes=5):
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        crop_init = (info['lulc'] == CROPLAND_CLASS).sum()
        done, total_reward, actions = False, 0, []
        while not done:
            action = policy_fn(obs, env)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            actions.append((int(action[0]), int(action[1])))
            done = done or trunc
        results.append({
            'reward': total_reward,
            'cropland_init': int(crop_init),
            'cropland_final': int(info['cropland_after']),
            'cropland_change': int(info['cropland_after'] - crop_init),
        })
    return {
        'mean_reward': float(np.mean([r['reward'] for r in results])),
        'std_reward': float(np.std([r['reward'] for r in results])),
        'mean_cropland_change': float(np.mean([r['cropland_change'] for r in results])),
        'std_cropland_change': float(np.std([r['cropland_change'] for r in results])),
    }


def random_policy(obs, env):
    return env.action_space.sample()


def greedy_policy(obs, env):
    for r in range(env.n_regions):
        if r not in env.intervened:
            return np.array([r, len(INTENSITY_LEVELS) - 1])
    return env.action_space.sample()


def train_one(seed, env_factory, timesteps, out_dir, label='ppo'):
    os.makedirs(out_dir, exist_ok=True)
    eval_path = os.path.join(out_dir, f'{label}_eval_seed{seed}.json')
    if os.path.exists(eval_path):
        print(f'    [{label} seed {seed}] exists, skip')
        with open(eval_path) as f:
            return json.load(f)

    env = env_factory()
    env_mon = Monitor(env)

    model = PPO(
        'MlpPolicy', env_mon,
        learning_rate=3e-4,
        n_steps=env.max_steps * 10,
        batch_size=50,
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

    model.save(os.path.join(out_dir, f'{label}_model_seed{seed}.zip'))

    eval_env = env_factory()
    def ppo_fn(obs, e):
        a, _ = model.predict(obs, deterministic=True)
        return a

    metrics = evaluate(eval_env, ppo_fn)
    result = {'seed': seed, 'label': label, 'training_time_s': train_time,
              'timesteps': timesteps, **metrics}

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'    [{label} seed {seed}] reward={result["mean_reward"]:.1f}, '
          f'crop={result["mean_cropland_change"]:.0f}, {train_time:.0f}s')
    return result


def run_baselines(env_factory, out_dir, n_seeds=15):
    os.makedirs(out_dir, exist_ok=True)
    for method, fn in [('random', random_policy), ('greedy', greedy_policy)]:
        print(f'  {method}:')
        for seed in range(n_seeds):
            eval_path = os.path.join(out_dir, f'{method}_eval_seed{seed}.json')
            if os.path.exists(eval_path):
                print(f'    [{method} seed {seed}] exists')
                continue
            env = env_factory()
            np.random.seed(seed)
            metrics = evaluate(env, fn)
            result = {'seed': seed, 'label': method, **metrics}
            with open(eval_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'    [{method} seed {seed}] reward={result["mean_reward"]:.1f}, '
                  f'crop={result["mean_cropland_change"]:.0f}')


def summarize(out_dir, label):
    evals = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.json') and 'eval' in f:
            with open(os.path.join(out_dir, f)) as fh:
                evals.append(json.load(fh))
    if not evals:
        return
    by_method = {}
    for e in evals:
        m = e.get('label', 'unknown')
        by_method.setdefault(m, []).append(e)
    print(f'\n  === {label} Summary ===')
    for m, es in sorted(by_method.items()):
        rewards = [e['mean_reward'] for e in es]
        crops = [e['mean_cropland_change'] for e in es]
        print(f'  {m:>8}: reward={np.mean(rewards):+.1f}+-{np.std(rewards):.1f}, '
              f'crop={np.mean(crops):+.0f}+-{np.std(crops):.0f} ({len(es)} seeds)')


# ============================================================
# Experiment runners
# ============================================================

def exp_county(n_seeds=15, timesteps=50_000):
    """Experiment 1: County-level 500m."""
    print('\n' + '='*60)
    print('EXPERIMENT 1: County-level (500m, K=50)')
    print('='*60)

    data_dir = os.path.join(SCRIPT_DIR, 'data')
    out_dir = os.path.join(RESULTS_DIR, 'county')
    ldn, decoder = load_ldn_model(), load_lulc_decoder()

    def factory():
        return make_env(data_dir, 'bishan_emb_2020.npy', 'bishan_context.npy',
                        n_regions=50, ldn=ldn, decoder=decoder)

    run_baselines(factory, out_dir, n_seeds)
    print('  ppo:')
    for seed in range(n_seeds):
        train_one(seed, factory, timesteps, out_dir)
        if (seed + 1) % 5 == 0:
            sync_to_drive()

    summarize(out_dir, 'County')
    sync_to_drive()


def exp_village(n_seeds=15, timesteps=50_000):
    """Experiment 2: Village-level 10m (Banzhucun 2017)."""
    print('\n' + '='*60)
    print('EXPERIMENT 2: Village-level (10m, K=300, Banzhucun 2017)')
    print('='*60)

    data_dir = os.path.join(SCRIPT_DIR, 'data', 'village')
    out_dir = os.path.join(RESULTS_DIR, 'village')
    ldn, decoder = load_ldn_model(), load_lulc_decoder()

    def factory():
        return make_env(data_dir, 'village_emb_2017_50m.npy', 'village_context_50m.npy',
                        n_regions=100, ldn=ldn, decoder=decoder)

    run_baselines(factory, out_dir, n_seeds)
    print('  ppo:')
    for seed in range(n_seeds):
        train_one(seed, factory, timesteps, out_dir)
        if (seed + 1) % 5 == 0:
            sync_to_drive()

    summarize(out_dir, 'Village')
    sync_to_drive()


def exp_transfer(n_seeds=15):
    """Experiment 3: Transfer Banzhucun -> Heping (zero-shot)."""
    print('\n' + '='*60)
    print('EXPERIMENT 3: Transfer (Banzhucun -> Heping, zero-shot)')
    print('='*60)

    village_dir = os.path.join(SCRIPT_DIR, 'data', 'village')
    heping_dir = os.path.join(SCRIPT_DIR, 'data', 'heping')
    train_dir = os.path.join(RESULTS_DIR, 'village')
    out_dir = os.path.join(RESULTS_DIR, 'transfer_heping')
    os.makedirs(out_dir, exist_ok=True)

    ldn, decoder = load_ldn_model(), load_lulc_decoder()

    # Load Heping env
    def heping_factory():
        return make_env(heping_dir, 'heping_emb_2017_50m.npy', 'heping_context_50m.npy',
                        n_regions=100, ldn=ldn, decoder=decoder)

    # Baselines on Heping
    run_baselines(heping_factory, out_dir, n_seeds)

    # Transfer: load Banzhucun-trained models, evaluate on Heping
    print('  transfer:')
    for seed in range(n_seeds):
        eval_path = os.path.join(out_dir, f'transfer_eval_seed{seed}.json')
        if os.path.exists(eval_path):
            print(f'    [transfer seed {seed}] exists')
            continue

        model_path = os.path.join(train_dir, f'ppo_model_seed{seed}.zip')
        if not os.path.exists(model_path):
            print(f'    [transfer seed {seed}] no trained model, skip')
            continue

        heping_env = heping_factory()
        model = PPO.load(model_path, env=heping_env)

        def transfer_fn(obs, e):
            a, _ = model.predict(obs, deterministic=True)
            return a

        metrics = evaluate(heping_env, transfer_fn)
        result = {'seed': seed, 'label': 'transfer', **metrics}
        with open(eval_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'    [transfer seed {seed}] reward={result["mean_reward"]:.1f}, '
              f'crop={result["mean_cropland_change"]:.0f}')

    summarize(out_dir, 'Transfer (Banzhucun->Heping)')
    sync_to_drive()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['county', 'village', 'transfer', 'all'],
                        default='all')
    parser.add_argument('--n_seeds', type=int, default=15)
    parser.add_argument('--timesteps', type=int, default=50_000)
    args = parser.parse_args()

    t0 = time.time()

    if args.exp in ('county', 'all'):
        exp_county(args.n_seeds, args.timesteps)

    if args.exp in ('village', 'all'):
        exp_village(args.n_seeds, args.timesteps)

    if args.exp in ('transfer', 'all'):
        exp_transfer(args.n_seeds)

    elapsed = time.time() - t0
    print(f'\n\nTotal elapsed: {elapsed/3600:.1f} hours')
    sync_to_drive()


if __name__ == '__main__':
    main()
