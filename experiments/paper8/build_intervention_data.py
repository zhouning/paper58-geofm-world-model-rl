# -*- coding: utf-8 -*-
"""
Paper 8 v2 Step 1: Build intervention training data.

Trains a BlockFeatureToEmbedding encoder that maps 17-dim block features
to 64-dim GeoFM embeddings. Then uses it to convert Paper 7's 12,000
trajectory transitions into embedding-space transitions.

Usage:
    python paper8/build_intervention_data.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER7_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper7')
PAPER8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PAPER8_DIR, 'data')


class BlockFeatureEncoder(nn.Module):
    """Maps 17-dim block features to 64-dim embedding space."""

    def __init__(self, in_dim=17, emb_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class BlockEmbeddingDataset(Dataset):
    """Dataset of (block_features, embedding) pairs for encoder training."""

    def __init__(self, block_features, embeddings):
        self.features = torch.tensor(block_features, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.embeddings[idx]


def train_encoder(block_features, embeddings, epochs=200, lr=1e-3, val_split=0.1):
    """Train BlockFeatureEncoder on (features, embedding) pairs.

    Args:
        block_features: (N, 17) initial block features from trajectory step 0
        embeddings: (N, 64) GeoFM embeddings

    Returns:
        trained encoder, training history
    """
    N = len(block_features)
    n_val = int(N * val_split)
    perm = np.random.permutation(N)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = BlockEmbeddingDataset(block_features[train_idx], embeddings[train_idx])
    val_ds = BlockEmbeddingDataset(block_features[val_idx], embeddings[val_idx])
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    encoder = BlockFeatureEncoder(in_dim=block_features.shape[1], emb_dim=embeddings.shape[1])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    cos_sim = nn.CosineSimilarity(dim=1)

    history = {'train_loss': [], 'val_loss': [], 'val_cosine': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        encoder.train()
        train_loss = 0
        for bf, emb in train_dl:
            pred = encoder(bf)
            loss = nn.functional.mse_loss(pred, emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bf)
        train_loss /= len(train_ds)

        # Validate
        encoder.eval()
        val_loss, val_cos = 0, 0
        with torch.no_grad():
            for bf, emb in val_dl:
                pred = encoder(bf)
                val_loss += nn.functional.mse_loss(pred, emb).item() * len(bf)
                val_cos += cos_sim(pred, emb).sum().item()
        val_loss /= len(val_ds)
        val_cos /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cosine'].append(val_cos)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f'  Epoch {epoch+1}: train_loss={train_loss:.6f}, '
                  f'val_loss={val_loss:.6f}, val_cosine={val_cos:.4f}')

    encoder.load_state_dict(best_state)
    return encoder, history


def build_intervention_dataset(encoder, traj_dir, block_embeddings):
    """Convert Paper 7 trajectories to compact embedding-space transitions.

    Only stores the selected block's embedding delta (not full 2600×64 grid),
    plus global features and reward. This reduces data from ~14GB to ~10MB.

    For each trajectory step:
      - Map selected block's current features → embedding via encoder
      - Map selected block's next features → embedding via encoder
      - Compute delta = next_emb - current_emb
      - Record: (action, selected_block_emb, emb_delta, reward, global_features)
    """
    encoder.eval()
    all_data = {
        'actions': [],           # (T,) int
        'selected_emb': [],      # (T, 64) current embedding of selected block
        'emb_delta': [],         # (T, 64) embedding change of selected block
        'rewards': [],           # (T,) float
        'global_features': [],   # (T, 12) float
        'next_global': [],       # (T, 12) float
        'dones': [],             # (T,) bool
    }

    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    for fname in files:
        data = np.load(os.path.join(traj_dir, fname))
        bf = data['block_features'].astype(np.float32)    # (T, N, 17)
        nbf = data['next_block_features'].astype(np.float32)
        actions = data['actions']
        rewards = data['rewards']
        gf = data['global_features']
        ngf = data['next_global_features'] if 'next_global_features' in data else gf
        dones = data['dones']

        T = len(actions)

        with torch.no_grad():
            for i in range(T):
                a = actions[i]
                cur_feat = torch.tensor(bf[i, a:a+1], dtype=torch.float32)  # (1, 17)
                nxt_feat = torch.tensor(nbf[i, a:a+1], dtype=torch.float32)
                cur_emb = encoder(cur_feat).numpy()[0]  # (64,)
                nxt_emb = encoder(nxt_feat).numpy()[0]

                all_data['actions'].append(a)
                all_data['selected_emb'].append(cur_emb)
                all_data['emb_delta'].append(nxt_emb - cur_emb)
                all_data['rewards'].append(rewards[i])
                all_data['global_features'].append(gf[i])
                all_data['next_global'].append(ngf[i] if i < len(ngf) else gf[i])
                all_data['dones'].append(dones[i])

        print(f'  {fname}: {T} transitions encoded')

    result = {k: np.array(v) for k, v in all_data.items()}
    result['n_blocks'] = bf.shape[1]
    result['emb_dim'] = 64
    result['k_global'] = gf.shape[-1]
    return result


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load static data
    block_emb = np.load(os.path.join(PAPER7_DIR, 'data', 'block_geofm_embeddings.npy'))
    print(f'Block embeddings: {block_emb.shape}')

    # Get initial block features from first trajectory step
    traj_dir = os.path.join(PAPER7_DIR, 'trajectories')
    first_traj = np.load(os.path.join(traj_dir, sorted(os.listdir(traj_dir))[0]))
    init_bf = first_traj['block_features'][0].astype(np.float32)  # (2600, 17)
    print(f'Initial block features: {init_bf.shape}')

    # Step 1: Train encoder
    print('\n=== Step 1: Train BlockFeatureEncoder ===')
    encoder, history = train_encoder(init_bf, block_emb)
    print(f'Best val cosine: {max(history["val_cosine"]):.4f}')

    # Save encoder
    encoder_path = os.path.join(DATA_DIR, 'block_feature_encoder.pt')
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'in_dim': 17,
        'emb_dim': 64,
        'val_cosine': max(history['val_cosine']),
    }, encoder_path)
    print(f'Saved encoder: {encoder_path}')

    # Step 2: Build intervention dataset
    print('\n=== Step 2: Build intervention dataset ===')
    dataset = build_intervention_dataset(encoder, traj_dir, block_emb)

    # Save
    out_path = os.path.join(DATA_DIR, 'intervention_transitions.npz')
    np.savez_compressed(out_path, **{k: v for k, v in dataset.items()})
    fsize = os.path.getsize(out_path) / 1024 / 1024
    print(f'\nSaved: {out_path} ({fsize:.1f} MB)')
    print(f'  Transitions: {len(dataset["actions"])}')
    print(f'  Embedding dim: {dataset["emb_dim"]}')
    print(f'  Blocks: {dataset["n_blocks"]}')

    # Verify: check embedding deltas make sense
    print('\n=== Verification ===')
    deltas = dataset['emb_delta']
    delta_norms = np.linalg.norm(deltas, axis=1)
    print(f'Embedding delta norm: mean={delta_norms.mean():.4f}, std={delta_norms.std():.4f}')
    print(f'  min={delta_norms.min():.4f}, max={delta_norms.max():.4f}')
    print(f'  >0.01: {(delta_norms > 0.01).sum()} / {len(delta_norms)} '
          f'({100*(delta_norms > 0.01).mean():.1f}%)')


if __name__ == '__main__':
    main()
