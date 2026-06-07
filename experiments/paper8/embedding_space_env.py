# -*- coding: utf-8 -*-
"""
Paper 8: EmbeddingSpaceEnv — Gymnasium environment operating in 64-dim
AlphaEarth embedding space with LatentDynamicsNet as transition model.

State:  z_t in R^{64 x H x W}  (flattened for obs space)
Action: MultiDiscrete([K, 5]) — K spatial regions x 5 scenarios
Transition: LatentDynamicsNet.forward(z_t, s_blended, context)
Reward: LULC change + spatial coherence (see embedding_reward.py)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn.functional as F

sys.path.insert(0, 'D:/adk/data_agent')
# Also support Colab layout (world_model.py in same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Scenario names matching world_model.py
SCENARIO_NAMES = [
    "urban_sprawl",            # 0
    "ecological_restoration",  # 1
    "agricultural_intensification",  # 2
    "climate_adaptation",      # 3
    "baseline",                # 4
]
# Intensity levels for baseline-only mode (Step A from design doc)
INTENSITY_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
SCENARIO_DIM = 16
CROPLAND_CLASS = 7
TREE_CLASS = 2


def load_ldn_model():
    """Load LatentDynamicsNet from saved weights."""
    from world_model import _build_model, Z_DIM, SCENARIO_DIM, N_CONTEXT
    # Try multiple weight paths (local Windows / Colab)
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'latent_dynamics_v1.pt'),
        'D:/adk/data_agent/weights/latent_dynamics_v1.pt',
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = _build_model(
        ckpt.get('z_dim', Z_DIM),
        ckpt.get('scenario_dim', SCENARIO_DIM),
        ckpt.get('n_context', N_CONTEXT),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_lulc_decoder():
    """Load LULC decoder (LogisticRegression)."""
    import joblib
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'lulc_decoder_v1.pkl'),
        'D:/adk/data_agent/weights/lulc_decoder_v1.pkl',
    ]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    return joblib.load(path)


def encode_scenario(name):
    """Encode scenario name to [1, 16] tensor."""
    vec = np.zeros(SCENARIO_DIM, dtype=np.float32)
    idx = SCENARIO_NAMES.index(name)
    vec[idx] = 1.0
    return torch.tensor(vec).unsqueeze(0)


def partition_regions(embedding_grid, n_regions=50):
    """K-means clustering on embedding grid to define spatial regions.

    Args:
        embedding_grid: [H, W, 64] numpy array
        n_regions: number of clusters

    Returns:
        region_map: [H, W] int array — region ID per pixel
        region_centers: [K, 64] cluster centers
    """
    from sklearn.cluster import KMeans
    H, W, D = embedding_grid.shape
    flat = embedding_grid.reshape(-1, D)
    km = KMeans(n_clusters=n_regions, random_state=42, n_init=5, max_iter=100)
    labels = km.fit_predict(flat)
    return labels.reshape(H, W), km.cluster_centers_


def decode_lulc(embedding_grid, decoder):
    """Decode embedding grid to LULC class grid.

    Args:
        embedding_grid: [H, W, 64] or [B, 64, H, W] tensor/array
        decoder: sklearn LogisticRegression

    Returns:
        lulc: [H, W] int array
    """
    if isinstance(embedding_grid, torch.Tensor):
        arr = embedding_grid.detach().cpu().numpy()
        if arr.ndim == 4:  # [B, 64, H, W] -> [H, W, 64]
            arr = arr[0].transpose(1, 2, 0)
    else:
        arr = embedding_grid
    H, W = arr.shape[:2]
    return decoder.predict(arr.reshape(-1, 64)).reshape(H, W)


class EmbeddingSpaceEnv(gym.Env):
    """Gymnasium environment operating in AlphaEarth embedding space.

    The LatentDynamicsNet serves as the transition model — this IS the
    "Dreaming in Embedding Space" concept from Paper 8.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        embedding_grid,      # [H, W, 64] initial embeddings (e.g. 2020)
        terrain_context,     # [2, H, W] DEM + slope
        ldn_model=None,      # LatentDynamicsNet (loaded if None)
        decoder=None,        # LULC decoder (loaded if None)
        n_regions=50,        # K-means region count
        max_steps=5,         # planning horizon (years)
        cropland_weight=1.0,
        coherence_weight=0.5,
        forest_penalty=0.3,
    ):
        super().__init__()

        self.H, self.W, self.D = embedding_grid.shape
        self.initial_emb = embedding_grid.copy()
        self.terrain_ctx = terrain_context.copy()
        self.max_steps = max_steps
        self.cropland_weight = cropland_weight
        self.coherence_weight = coherence_weight
        self.forest_penalty = forest_penalty

        # Load models
        self.ldn = ldn_model if ldn_model is not None else load_ldn_model()
        self.decoder = decoder if decoder is not None else load_lulc_decoder()

        # Partition into regions
        self.n_regions = n_regions
        self.region_map, self.region_centers = partition_regions(
            embedding_grid, n_regions
        )
        # Precompute region pixel indices
        self.region_pixels = {}
        for k in range(n_regions):
            mask = self.region_map == k
            self.region_pixels[k] = np.argwhere(mask)  # [N_k, 2]

        self.n_scenarios = len(INTENSITY_LEVELS)

        # Spaces
        obs_dim = self.n_regions * self.D + 4  # region embeddings + global features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.n_regions, self.n_scenarios])

        # State
        self.current_emb = None
        self.step_count = 0
        self.intervened = None  # track which regions have been intervened

    def _get_region_embeddings(self):
        """Compute mean embedding per region from current state."""
        region_embs = np.zeros((self.n_regions, self.D), dtype=np.float32)
        for k in range(self.n_regions):
            pixels = self.region_pixels[k]
            if len(pixels) > 0:
                region_embs[k] = self.current_emb[pixels[:, 0], pixels[:, 1]].mean(axis=0)
        return region_embs

    def _get_global_features(self):
        """Compute global state features."""
        lulc = decode_lulc(self.current_emb, self.decoder)
        total = self.H * self.W
        cropland_frac = (lulc == CROPLAND_CLASS).sum() / total
        tree_frac = (lulc == TREE_CLASS).sum() / total
        step_frac = self.step_count / self.max_steps
        budget_frac = 1.0 - step_frac
        return np.array([cropland_frac, tree_frac, step_frac, budget_frac],
                        dtype=np.float32)

    def _get_obs(self):
        """Flatten region embeddings + global features into observation."""
        region_embs = self._get_region_embeddings()
        global_feats = self._get_global_features()
        return np.concatenate([region_embs.ravel(), global_feats])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_emb = self.initial_emb.copy()
        self.step_count = 0
        self.intervened = set()
        obs = self._get_obs()
        info = {'lulc': decode_lulc(self.current_emb, self.decoder)}
        return obs, info

    def step(self, action):
        region_id, intensity_id = int(action[0]), int(action[1])
        intensity = INTENSITY_LEVELS[intensity_id]

        # Decode LULC before
        lulc_before = decode_lulc(self.current_emb, self.decoder)

        # Use baseline scenario with intensity scaling
        baseline_vec = encode_scenario("baseline")  # [1, 16]

        # Convert current embedding to tensor [1, 64, H, W]
        z_t = torch.tensor(
            self.current_emb.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0)

        ctx = torch.tensor(self.terrain_ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            z_next_full = self.ldn(z_t, baseline_vec, ctx)

        # Compute delta and apply intensity scaling to selected region (vectorized)
        delta = z_next_full - z_t
        z_next = z_next_full.clone()  # baseline transition everywhere

        # Override selected region with intensity-scaled delta
        pixels = self.region_pixels[region_id]
        rows, cols = pixels[:, 0], pixels[:, 1]
        z_next[0, :, rows, cols] = z_t[0, :, rows, cols] + intensity * delta[0, :, rows, cols]

        # Normalize to unit hypersphere
        z_next = F.normalize(z_next, p=2, dim=1)

        # Update state
        self.current_emb = z_next[0].permute(1, 2, 0).numpy()
        self.step_count += 1
        self.intervened.add(region_id)

        # Decode LULC after
        lulc_after = decode_lulc(self.current_emb, self.decoder)

        # Compute reward
        reward = self._compute_reward(lulc_before, lulc_after)

        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            'lulc': lulc_after,
            'region_id': region_id,
            'intensity': intensity,
            'cropland_before': (lulc_before == CROPLAND_CLASS).sum(),
            'cropland_after': (lulc_after == CROPLAND_CLASS).sum(),
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, lulc_before, lulc_after):
        """Three-layer reward: LULC change + coherence + forest penalty."""
        # Layer 1: Cropland change (positive = more cropland = good)
        crop_change = (lulc_after == CROPLAND_CLASS).sum() - (lulc_before == CROPLAND_CLASS).sum()

        # Layer 1b: Forest change (negative = less forest = penalty)
        forest_change = (lulc_after == TREE_CLASS).sum() - (lulc_before == TREE_CLASS).sum()

        # Layer 2: Spatial coherence of cropland pixels
        coherence = self._compute_coherence()

        reward = (
            self.cropland_weight * crop_change
            + self.coherence_weight * coherence
            - self.forest_penalty * max(0, -forest_change)
        )
        return float(reward)

    def _compute_coherence(self):
        """Mean cosine similarity between neighboring cropland pixels (vectorized)."""
        lulc = decode_lulc(self.current_emb, self.decoder)
        crop_mask = lulc == CROPLAND_CLASS
        if crop_mask.sum() < 2:
            return 0.0

        emb = self.current_emb  # [H, W, 64]
        # Compute neighbor similarities using shifted arrays (avoid Python loops)
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_normed = emb / norms

        total_sim = 0.0
        count = 0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Shifted crop mask overlap
            shifted_mask = np.zeros_like(crop_mask)
            src_r = slice(max(0, -di), self.H - max(0, di))
            src_c = slice(max(0, -dj), self.W - max(0, dj))
            dst_r = slice(max(0, di), self.H + min(0, di))
            dst_c = slice(max(0, dj), self.W + min(0, dj))
            shifted_mask[dst_r, dst_c] = crop_mask[src_r, src_c]

            both = crop_mask & shifted_mask
            if not both.any():
                continue

            # Cosine similarity at overlapping positions
            shifted_emb = np.zeros_like(emb_normed)
            shifted_emb[dst_r, dst_c] = emb_normed[src_r, src_c]

            dots = (emb_normed * shifted_emb).sum(axis=-1)  # [H, W]
            total_sim += dots[both].sum()
            count += both.sum()

        return float(total_sim / max(count, 1))

    def action_masks(self):
        """Boolean mask [n_regions * n_scenarios]. Mask out already-intervened regions."""
        mask = np.ones(self.n_regions * self.n_scenarios, dtype=bool)
        for k in self.intervened:
            mask[k * self.n_scenarios:(k + 1) * self.n_scenarios] = False
        return mask


if __name__ == '__main__':
    # Quick smoke test
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    emb = np.load(os.path.join(data_dir, 'bishan_emb_2020.npy'))
    ctx = np.load(os.path.join(data_dir, 'bishan_context.npy'))

    print(f'Building EmbeddingSpaceEnv: emb={emb.shape}, ctx={ctx.shape}')
    env = EmbeddingSpaceEnv(emb, ctx, n_regions=50, max_steps=5)
    print(f'Obs space: {env.observation_space.shape}')
    print(f'Action space: {env.action_space}')

    obs, info = env.reset()
    print(f'Reset: obs shape={obs.shape}, cropland={info["lulc"].flatten().tolist().count(7)}')

    # Run 3 random steps
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f'Step {step+1}: action=(region={action[0]}, intensity={INTENSITY_LEVELS[action[1]]}), '
              f'reward={reward:.4f}, cropland={info["cropland_after"]}, done={done}')

    print('Smoke test passed!')
