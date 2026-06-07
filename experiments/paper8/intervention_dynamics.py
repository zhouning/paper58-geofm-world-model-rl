# -*- coding: utf-8 -*-
"""
Paper 8 v2 Step 2: InterventionDynamicsNet — learns intervention effects in embedding space.

Predicts how block embeddings change when a block is selected for investment.
Trained on Paper 7's 12,000 trajectory transitions mapped to embedding space.

Usage:
    python paper8/intervention_dynamics.py           # train
    python paper8/intervention_dynamics.py --verify   # verify only
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PAPER8_DIR, 'data')


class InterventionDataset(Dataset):
    """Compact dataset: only stores selected block's embedding + delta."""

    def __init__(self, data_path, indices=None):
        data = np.load(data_path)
        self.n_blocks = int(data['n_blocks'])
        self.emb_dim = int(data['emb_dim'])
        self.k_global = int(data['k_global'])

        if indices is not None:
            self.selected_emb = data['selected_emb'][indices]     # (T, 64)
            self.emb_delta = data['emb_delta'][indices]           # (T, 64)
            self.actions = data['actions'][indices]                # (T,)
            self.rewards = data['rewards'][indices]                # (T,)
            self.global_features = data['global_features'][indices]  # (T, 12)
        else:
            self.selected_emb = data['selected_emb']
            self.emb_delta = data['emb_delta']
            self.actions = data['actions']
            self.rewards = data['rewards']
            self.global_features = data['global_features']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.selected_emb[idx], dtype=torch.float32),    # (64,)
            torch.tensor(self.global_features[idx], dtype=torch.float32), # (12,)
            torch.tensor(self.actions[idx], dtype=torch.long),            # scalar
            torch.tensor(self.emb_delta[idx], dtype=torch.float32),      # (64,)
            torch.tensor(self.rewards[idx], dtype=torch.float32),         # scalar
        )


class InterventionDynamicsNet(nn.Module):
    """Predicts embedding delta for a selected block given investment action.

    Compact version: operates on single-block embeddings, not full grid.
    For full-grid inference, call forward() per selected block.
    """

    def __init__(self, n_blocks, emb_dim=64, k_global=12, hidden_dim=256):
        super().__init__()
        self.n_blocks = n_blocks
        self.emb_dim = emb_dim
        self.k_global = k_global

        # Action embedding
        self.action_emb = nn.Embedding(n_blocks, 64)

        # Global encoder
        self.global_enc = nn.Sequential(
            nn.Linear(k_global, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Context = selected_emb(64) + action_emb(64) + global_enc(64) = 192
        ctx_dim = 192

        # Delta head: predict embedding residual for selected block
        self.delta_head = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, selected_emb, global_features, action):
        """
        Args:
            selected_emb: (B, emb_dim) embedding of selected block
            global_features: (B, k_global)
            action: (B,) block index

        Returns:
            emb_delta: (B, emb_dim) predicted embedding change
            reward: (B,)
        """
        h_action = self.action_emb(action)        # (B, 64)
        h_global = self.global_enc(global_features)  # (B, 64)
        ctx = torch.cat([selected_emb, h_action, h_global], dim=1)  # (B, 192)

        delta = self.delta_head(ctx)          # (B, emb_dim)
        reward = self.reward_head(ctx).squeeze(-1)  # (B,)
        return delta, reward


def train_dynamics(data_path, epochs=200, lr=1e-3, batch_size=128, val_split=0.1):
    """Train InterventionDynamicsNet."""
    full_data = np.load(data_path)
    n_total = len(full_data['actions'])
    n_blocks = int(full_data['n_blocks'])
    emb_dim = int(full_data['emb_dim'])
    k_global = int(full_data['k_global'])

    perm = np.random.permutation(n_total)
    n_val = int(n_total * val_split)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = InterventionDataset(data_path, train_idx)
    val_ds = InterventionDataset(data_path, val_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = InterventionDynamicsNet(n_blocks, emb_dim, k_global)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    cos_sim = nn.CosineSimilarity(dim=1)

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    history = {'train_loss': [], 'val_loss': [], 'val_cosine': [], 'val_reward_mse': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sel_emb, gf, action, true_delta, reward in train_dl:
            pred_delta, pred_r = model(sel_emb, gf, action)
            loss_emb = nn.functional.mse_loss(pred_delta, true_delta)
            loss_reward = nn.functional.mse_loss(pred_r, reward)
            loss = loss_emb + 0.1 * loss_reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(sel_emb)
        total_loss /= len(train_ds)

        model.eval()
        val_loss, val_cos, val_rmse, n_val_samples = 0, 0, 0, 0
        with torch.no_grad():
            for sel_emb, gf, action, true_delta, reward in val_dl:
                pred_delta, pred_r = model(sel_emb, gf, action)
                B = len(sel_emb)
                val_loss += nn.functional.mse_loss(pred_delta, true_delta).item() * B

                # Cosine sim between predicted and true next_emb
                pred_next = sel_emb + pred_delta
                true_next = sel_emb + true_delta
                val_cos += cos_sim(pred_next, true_next).sum().item()

                val_rmse += nn.functional.mse_loss(pred_r, reward).item() * B
                n_val_samples += B

        val_loss /= n_val_samples
        val_cos /= n_val_samples
        val_rmse /= n_val_samples

        history['train_loss'].append(total_loss)
        history['val_loss'].append(val_loss)
        history['val_cosine'].append(val_cos)
        history['val_reward_mse'].append(val_rmse)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f'  Epoch {epoch+1}: train={total_loss:.6f}, val={val_loss:.6f}, '
                  f'cos={val_cos:.4f}, reward_mse={val_rmse:.4f}')

    model.load_state_dict(best_state)
    return model, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    data_path = os.path.join(DATA_DIR, 'intervention_transitions.npz')
    model_path = os.path.join(DATA_DIR, 'intervention_dynamics.pt')

    if not args.verify:
        print('=== Training InterventionDynamicsNet ===')
        model, history = train_dynamics(data_path, epochs=100, batch_size=32)

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_blocks': model.n_blocks,
            'emb_dim': model.emb_dim,
            'k_global': model.k_global,
            'best_val_loss': min(history['val_loss']),
            'best_val_cosine': max(history['val_cosine']),
        }, model_path)
        print(f'\nSaved: {model_path}')
        print(f'Best val cosine: {max(history["val_cosine"]):.4f}')
        print(f'Best val loss: {min(history["val_loss"]):.6f}')

    # Verify
    print('\n=== Verification ===')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = InterventionDynamicsNet(
        ckpt['n_blocks'], ckpt['emb_dim'], ckpt['k_global'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model: cosine={ckpt["best_val_cosine"]:.4f}, loss={ckpt["best_val_loss"]:.6f}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')


if __name__ == '__main__':
    main()
