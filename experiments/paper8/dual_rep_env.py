# -*- coding: utf-8 -*-
"""
Paper 8 v3: Dual-Representation InterventionEnv.

State = block_features (N×17) + block_embeddings (N×64) = N×81 + global (12)
Train on Bishan with both representations available.
Transfer to new regions: only embeddings available, features zero-filled.

This combines Paper 7's proven optimization with Paper 8's transfer capability.
"""

import os
import sys
import json
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAPER7_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper7')
PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))


class DualRepEnv(gym.Env):
    """Block-level RL environment with dual representation.

    Observation per block = [17-dim features | 64-dim GeoFM embedding] = 81-dim.
    Uses Paper 7's TransitionModel for feature dynamics +
    InterventionDynamicsNet for embedding dynamics.

    At transfer time, features can be zeroed out — policy still has embeddings.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model,        # Paper 7 TransitionModel
        initial_block_features,  # (N, 17)
        initial_global_features, # (12,)
        block_embeddings,        # (N, 64) static GeoFM
        intervention_dynamics=None,  # optional: update embeddings too
        max_steps=100,
        feature_dropout=0.0,     # probability of zeroing features (for transfer robustness)
    ):
        super().__init__()

        self.n_blocks = initial_block_features.shape[0]
        self.k_feat = initial_block_features.shape[1]   # 17
        self.k_emb = block_embeddings.shape[1]           # 64
        self.k_block = self.k_feat + self.k_emb          # 81
        self.k_global = initial_global_features.shape[0]  # 12
        self.max_steps = max_steps
        self.feature_dropout = feature_dropout

        self.init_features = initial_block_features.astype(np.float32)
        self.init_global = initial_global_features.astype(np.float32)
        self.static_embeddings = block_embeddings.astype(np.float32)

        self.tm = transition_model  # Paper 7 TransitionModel
        self.idm = intervention_dynamics  # InterventionDynamicsNet (optional)

        # Obs = N×81 + 12
        obs_dim = self.n_blocks * self.k_block + self.k_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_blocks)

        self.block_features = None
        self.block_embs = None
        self.global_features = None
        self.step_count = 0

    def _get_obs(self):
        # Concatenate features + embeddings per block, then global
        feat = self.block_features  # (N, 17)
        emb = self.block_embs       # (N, 64)

        # Optional: dropout features for transfer robustness training
        if self.feature_dropout > 0 and np.random.random() < self.feature_dropout:
            feat = np.zeros_like(feat)

        combined = np.concatenate([feat, emb], axis=1)  # (N, 81)
        return np.concatenate([combined.ravel(), self.global_features]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.block_features = self.init_features.copy()
        self.block_embs = self.static_embeddings.copy()
        self.global_features = self.init_global.copy()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)

        # Use Paper 7's TransitionModel for feature dynamics
        bf_t = torch.tensor(self.block_features[np.newaxis], dtype=torch.float32)
        gf_t = torch.tensor(self.global_features[np.newaxis], dtype=torch.float32)
        act_t = torch.tensor([action], dtype=torch.long)

        with torch.no_grad():
            pred_bf, pred_gf, pred_reward = self.tm(bf_t, gf_t, act_t)

        self.block_features = pred_bf[0].numpy()
        self.global_features = pred_gf[0].numpy()
        reward = float(pred_reward[0].item())

        # Update embedding for selected block via InterventionDynamicsNet
        if self.idm is not None:
            sel_emb = torch.tensor(self.block_embs[action:action+1], dtype=torch.float32)
            with torch.no_grad():
                delta, _ = self.idm(sel_emb, gf_t, act_t)
            self.block_embs[action] += delta.numpy()[0]

        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        info = {'action': action, 'step': self.step_count}
        return self._get_obs(), reward, terminated, False, info

    def action_masks(self):
        # Same mask logic as Paper 7: feature[9] > threshold (swap potential)
        return self.block_features[:, 9] > 0.01


def load_paper7_tm():
    """Load Paper 7's trained TransitionModel."""
    from paper7.learned_env import TransitionModel
    path = os.path.join(PAPER7_DIR, 'models', 'transition_model.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = TransitionModel(
        n_blocks=int(ckpt['n_blocks']),
        k_block=int(ckpt['k_block']),
        k_global=int(ckpt['k_global']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_intervention_dynamics():
    """Load InterventionDynamicsNet."""
    from intervention_dynamics import InterventionDynamicsNet
    path = os.path.join(PAPER8_DIR, 'data', 'intervention_dynamics.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = InterventionDynamicsNet(ckpt['n_blocks'], ckpt['emb_dim'], ckpt['k_global'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def make_dual_env(feature_dropout=0.0):
    """Create DualRepEnv with Paper 7 data."""
    traj_dir = os.path.join(PAPER7_DIR, 'trajectories')
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(traj_dir, files[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]
    block_emb = np.load(os.path.join(PAPER7_DIR, 'data', 'block_geofm_embeddings.npy'))

    tm = load_paper7_tm()
    idm = load_intervention_dynamics()

    return DualRepEnv(tm, init_bf, init_gf, block_emb,
                      intervention_dynamics=idm,
                      max_steps=100,
                      feature_dropout=feature_dropout)


if __name__ == '__main__':
    print('Building DualRepEnv...')
    env = make_dual_env()
    print(f'Obs space: {env.observation_space.shape}')
    print(f'  = {env.n_blocks} blocks × {env.k_block} (17 feat + 64 emb) + {env.k_global} global')
    print(f'Action space: {env.action_space}')

    obs, _ = env.reset()
    print(f'Reset: obs shape={obs.shape}')

    total_reward = 0
    for step in range(5):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        print(f'  Step {step}: action={action}, reward={reward:+.4f}')

    print(f'Total reward (5 steps): {total_reward:+.4f}')
    print('Smoke test passed!')
