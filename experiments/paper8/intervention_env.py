# -*- coding: utf-8 -*-
"""
Paper 8 v2 Step 3: InterventionEnv — Gymnasium environment using
InterventionDynamicsNet for block-level optimization in embedding space.

State:  block embeddings (N×64) + global features (12)
Action: Discrete(N) — select which block to invest
Transition: InterventionDynamicsNet predicts embedding delta + reward
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PAPER8_DIR, 'data')


def load_intervention_model():
    """Load trained InterventionDynamicsNet."""
    from intervention_dynamics import InterventionDynamicsNet
    path = os.path.join(DATA_DIR, 'intervention_dynamics.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = InterventionDynamicsNet(
        ckpt['n_blocks'], ckpt['emb_dim'], ckpt['k_global'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_encoder():
    """Load trained BlockFeatureEncoder."""
    from build_intervention_data import BlockFeatureEncoder
    path = os.path.join(DATA_DIR, 'block_feature_encoder.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    enc = BlockFeatureEncoder(ckpt['in_dim'], ckpt['emb_dim'])
    enc.load_state_dict(ckpt['model_state_dict'])
    enc.eval()
    return enc


class InterventionEnv(gym.Env):
    """Block-level RL environment in embedding space.

    Uses InterventionDynamicsNet (trained on real intervention trajectories)
    instead of LDN (natural dynamics only). This enables actual optimization.

    Compatible with MaskablePPO + ParcelScoringPolicy from Paper 4/7.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_block_features,   # (N, 17) from trajectory data
        initial_global_features,  # (12,)
        block_embeddings,         # (N, 64) static GeoFM embeddings
        dynamics_model=None,
        encoder=None,
        max_steps=100,
        reward_scale=1.0,
    ):
        super().__init__()

        self.n_blocks = initial_block_features.shape[0]
        self.k_block = initial_block_features.shape[1]
        self.emb_dim = block_embeddings.shape[1]
        self.k_global = initial_global_features.shape[0]
        self.max_steps = max_steps
        self.reward_scale = reward_scale

        # Store initial state
        self.init_block_features = initial_block_features.copy()
        self.init_global_features = initial_global_features.copy()
        self.static_embeddings = block_embeddings.copy()

        # Load models
        self.dynamics = dynamics_model if dynamics_model is not None else load_intervention_model()
        self.encoder = encoder if encoder is not None else load_encoder()

        # Observation: flattened block embeddings + global features
        obs_dim = self.n_blocks * self.emb_dim + self.k_global
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_blocks)

        # State
        self.block_embs = None
        self.global_features = None
        self.step_count = 0
        self.invested = None

    def _encode_features(self, block_features):
        """Encode block features to embeddings via trained encoder."""
        with torch.no_grad():
            embs = self.encoder(torch.tensor(block_features, dtype=torch.float32))
        return embs.numpy()

    def _get_obs(self):
        return np.concatenate([
            self.block_embs.ravel(),
            self.global_features
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize embeddings from block features via encoder
        self.block_embs = self._encode_features(self.init_block_features)
        self.global_features = self.init_global_features.copy()
        self.step_count = 0
        self.invested = set()

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)

        # Get current embedding of selected block
        sel_emb = torch.tensor(self.block_embs[action:action+1], dtype=torch.float32)
        gf = torch.tensor(self.global_features[np.newaxis, :], dtype=torch.float32)
        act = torch.tensor([action], dtype=torch.long)

        # Predict delta and reward
        with torch.no_grad():
            delta, reward = self.dynamics(sel_emb, gf, act)

        # Update selected block embedding
        self.block_embs[action] += delta.numpy()[0]

        # Update global features (simple: increment step fraction, budget)
        self.step_count += 1
        self.invested.add(action)
        step_frac = self.step_count / self.max_steps
        self.global_features[3] = step_frac  # step_fraction index

        reward_val = float(reward.item()) * self.reward_scale

        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            'action': action,
            'emb_delta_norm': float(np.linalg.norm(delta.numpy())),
            'n_invested': len(self.invested),
        }

        return self._get_obs(), reward_val, terminated, truncated, info

    def action_masks(self):
        """Mask already-invested blocks (optional — can allow re-investment)."""
        mask = np.ones(self.n_blocks, dtype=bool)
        # Allow re-investment but track it
        return mask


if __name__ == '__main__':
    # Smoke test
    traj_dir = os.path.join(os.path.dirname(PAPER8_DIR), 'paper7', 'trajectories')
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(traj_dir, files[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]
    block_emb = np.load(os.path.join(os.path.dirname(PAPER8_DIR), 'paper7', 'data', 'block_geofm_embeddings.npy'))

    print(f'Building InterventionEnv: blocks={init_bf.shape[0]}, emb_dim={block_emb.shape[1]}')
    env = InterventionEnv(init_bf, init_gf, block_emb, max_steps=100)
    print(f'Obs space: {env.observation_space.shape}')
    print(f'Action space: {env.action_space}')

    obs, _ = env.reset()
    print(f'Reset: obs shape={obs.shape}')

    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        print(f'  Step {step}: action={action}, reward={reward:+.4f}, '
              f'delta_norm={info["emb_delta_norm"]:.4f}')

    print(f'Total reward (5 steps): {total_reward:+.4f}')
    print('Smoke test passed!')
