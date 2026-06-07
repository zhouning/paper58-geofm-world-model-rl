# -*- coding: utf-8 -*-
"""
Paper 8 v3: Train on DualRepEnv, evaluate on real CountyLevelEnv.

The policy sees both features (17-dim) and embeddings (64-dim) per block.
At eval time on real env, we provide real features + static embeddings.

Usage:
    python paper8/train_dual_rep.py --n_seeds 3 --timesteps 50000   # quick test
    python paper8/train_dual_rep.py --n_seeds 15 --timesteps 100000  # full
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

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from parcel_scoring_policy import ParcelScoringPolicy
from dual_rep_env import make_dual_env, load_intervention_dynamics

PAPER7_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper7')
PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER8_DIR, 'results', 'dual_rep')


def evaluate_real_env(model_path, n_episodes=5):
    """Evaluate on real CountyLevelEnv with dual representation.

    Real env provides features (17-dim). We append static GeoFM embeddings (64-dim)
    to create the 81-dim per-block observation the policy expects.
    """
    block_emb = np.load(os.path.join(PAPER7_DIR, 'data', 'block_geofm_embeddings.npy'))
    idm = load_intervention_dynamics()

    real_env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    n_blocks = real_env.n_blocks
    model = MaskablePPO.load(model_path)

    results = []
    for ep in range(n_episodes):
        obs_real, _ = real_env.reset()
        emb_state = block_emb.copy()
        done, total_reward = False, 0

        while not done:
            # Build dual obs: real features + embedding state
            bf = obs_real[:n_blocks * K_BLOCK].reshape(n_blocks, K_BLOCK)
            gf = obs_real[n_blocks * K_BLOCK:]
            combined = np.concatenate([bf, emb_state], axis=1)  # (N, 81)
            dual_obs = np.concatenate([combined.ravel(), gf]).astype(np.float32)

            mask = real_env.action_masks()
            action, _ = model.predict(dual_obs, action_masks=mask, deterministic=True)
            action = int(action)

            obs_real, r, terminated, truncated, info = real_env.step(action)
            done = terminated or truncated
            total_reward += r

            # Update embedding state
            sel_emb = torch.tensor(emb_state[action:action+1], dtype=torch.float32)
            gf_t = torch.tensor(gf[np.newaxis, :], dtype=torch.float32)
            act_t = torch.tensor([action], dtype=torch.long)
            with torch.no_grad():
                delta, _ = idm(sel_emb, gf_t, act_t)
            emb_state[action] += delta.numpy()[0]

        results.append({
            'reward': total_reward,
            'slope_change_pct': info.get('slope_change_pct', 0),
            'cont_change': info.get('cont_change', 0),
        })

    return {
        'mean_reward': float(np.mean([r['reward'] for r in results])),
        'mean_slope': float(np.mean([r['slope_change_pct'] for r in results])),
        'mean_cont': float(np.mean([r['cont_change'] for r in results])),
    }


def train_one(seed, timesteps=100_000, feature_dropout=0.0):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    label = f'dropout{feature_dropout:.1f}' if feature_dropout > 0 else 'full'
    eval_path = os.path.join(RESULTS_DIR, f'{label}_eval_seed{seed}.json')
    if os.path.exists(eval_path):
        print(f'  [{label} seed {seed}] exists, skip')
        with open(eval_path) as f:
            return json.load(f)

    env = make_dual_env(feature_dropout=feature_dropout)
    env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy, env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(
            k_parcel=81,  # 17 features + 64 embeddings
            k_global=12,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    model_path = os.path.join(RESULTS_DIR, f'{label}_model_seed{seed}.zip')
    model.save(model_path)

    print(f'  [{label} seed {seed}] Training done ({train_time:.0f}s), evaluating...')
    real_metrics = evaluate_real_env(model_path)

    result = {
        'seed': seed, 'label': label,
        'feature_dropout': feature_dropout,
        'training_time_s': train_time,
        'timesteps': timesteps,
        **real_metrics,
    }

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  [{label} seed {seed}] slope={result["mean_slope"]:+.3f}%, '
          f'cont={result["mean_cont"]:+.4f}, time={train_time:.0f}s')
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Feature dropout rate for transfer robustness')
    args = parser.parse_args()

    print('='*60)
    print(f'Paper 8 v3: DualRepEnv (dropout={args.dropout})')
    print('='*60)

    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        r = train_one(seed, args.timesteps, args.dropout)
        results.append(r)

    slopes = [r['mean_slope'] for r in results]
    conts = [r['mean_cont'] for r in results]
    print(f'\n{"="*60}')
    print(f'Results ({len(results)} seeds):')
    print(f'  Slope: {np.mean(slopes):+.3f}% +- {np.std(slopes):.3f}%')
    print(f'  Cont:  {np.mean(conts):+.4f} +- {np.std(conts):.4f}')
    print(f'  Time:  {(time.time()-t0)/60:.1f} min')
    print(f'\n  References:')
    print(f'    Paper 7 no_cal:   -0.976% +- 0.129%')
    print(f'    Paper 7 with_cal: -1.102% +- 0.100%')
    print(f'    Paper 8 v2:       -0.028% +- 0.005%')


if __name__ == '__main__':
    main()
