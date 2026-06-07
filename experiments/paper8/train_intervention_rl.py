# -*- coding: utf-8 -*-
"""
Paper 8 v2 Step 4: Train RL on InterventionEnv, evaluate on real CountyLevelEnv.

Usage:
    python paper8/train_intervention_rl.py                  # 5 seeds quick test
    python paper8/train_intervention_rl.py --n_seeds 15     # full experiment
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
from intervention_env import InterventionEnv

PAPER7_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper7')
PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PAPER8_DIR, 'results', 'intervention')


def make_env():
    """Create InterventionEnv from Paper 7 trajectory data."""
    traj_dir = os.path.join(PAPER7_DIR, 'trajectories')
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(traj_dir, files[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]
    block_emb = np.load(os.path.join(PAPER7_DIR, 'data', 'block_geofm_embeddings.npy'))
    return InterventionEnv(init_bf, init_gf, block_emb, max_steps=100)


def evaluate_real_env(model_path, encoder=None, n_episodes=5):
    """Evaluate trained model on real CountyLevelEnv.

    Strategy C: maintain embedding state via InterventionDynamicsNet,
    don't re-encode features each step. Policy sees clean embedding evolution.
    """
    from intervention_env import load_intervention_model, load_encoder
    from intervention_dynamics import InterventionDynamicsNet

    dynamics = load_intervention_model()
    if encoder is None:
        encoder = load_encoder()

    real_env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    n_blocks = real_env.n_blocks

    # Load static GeoFM embeddings as initial state
    block_emb = np.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'paper7', 'data',
        'block_geofm_embeddings.npy'))

    model = MaskablePPO.load(model_path)

    results = []
    for ep in range(n_episodes):
        obs_real, _ = real_env.reset()
        # Initialize embedding state from static GeoFM (not from encoder)
        emb_state = block_emb.copy()  # (2600, 64)
        gf = obs_real[n_blocks * K_BLOCK:]  # global features from real env

        done, total_reward = False, 0
        while not done:
            # Build embedding obs for policy
            emb_obs = np.concatenate([emb_state.ravel(), gf]).astype(np.float32)

            # Get action
            mask = real_env.action_masks()
            action, _ = model.predict(emb_obs, action_masks=mask, deterministic=True)
            action = int(action)

            # Execute in real env
            obs_real, r, terminated, truncated, info = real_env.step(action)
            done = terminated or truncated
            total_reward += r

            # Update embedding state via dynamics model (not encoder)
            sel_emb = torch.tensor(emb_state[action:action+1], dtype=torch.float32)
            gf_t = torch.tensor(gf[np.newaxis, :], dtype=torch.float32)
            act_t = torch.tensor([action], dtype=torch.long)
            with torch.no_grad():
                delta, _ = dynamics(sel_emb, gf_t, act_t)
            emb_state[action] += delta.numpy()[0]

            # Update global features from real env
            gf = obs_real[n_blocks * K_BLOCK:]

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


def train_one(seed, timesteps=100_000):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_path = os.path.join(RESULTS_DIR, f'ppo_eval_seed{seed}.json')
    if os.path.exists(eval_path):
        print(f'  [seed {seed}] exists, skip')
        with open(eval_path) as f:
            return json.load(f)

    env = make_env()
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
            k_parcel=64,  # embedding dim per block
            k_global=12,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    model_path = os.path.join(RESULTS_DIR, f'ppo_model_seed{seed}.zip')
    model.save(model_path)

    # Evaluate on real env
    print(f'  [seed {seed}] Training done ({train_time:.0f}s), evaluating on real env...')
    real_metrics = evaluate_real_env(model_path)

    result = {
        'seed': seed,
        'training_time_s': train_time,
        'timesteps': timesteps,
        **real_metrics,
    }

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  [seed {seed}] slope={result["mean_slope"]:+.3f}%, '
          f'cont={result["mean_cont"]:+.4f}, time={train_time:.0f}s')
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--timesteps', type=int, default=100_000)
    args = parser.parse_args()

    print('='*60)
    print('Paper 8 v2: InterventionEnv RL Training')
    print('='*60)

    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        r = train_one(seed, args.timesteps)
        results.append(r)

    # Summary
    slopes = [r['mean_slope'] for r in results]
    conts = [r['mean_cont'] for r in results]
    rewards = [r['mean_reward'] for r in results]

    print(f'\n{"="*60}')
    print(f'Results ({len(results)} seeds):')
    print(f'  Slope: {np.mean(slopes):+.3f}% +- {np.std(slopes):.3f}%')
    print(f'  Cont:  {np.mean(conts):+.4f} +- {np.std(conts):.4f}')
    print(f'  Reward: {np.mean(rewards):+.1f} +- {np.std(rewards):.1f}')
    print(f'  Total time: {(time.time()-t0)/60:.1f} min')

    # Compare with Paper 7
    print(f'\n  Paper 7 reference (15 seeds):')
    print(f'    no_cal:   -0.976% +- 0.129%')
    print(f'    with_cal: -1.102% +- 0.100%')


if __name__ == '__main__':
    main()
