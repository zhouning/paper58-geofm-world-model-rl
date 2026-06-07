# -*- coding: utf-8 -*-
"""
Paper 8: Block Translator — maps embedding-space decisions to real-env block actions.

Given a (region_id, scenario_id) from EmbeddingSpaceEnv, translates to a
block_id for evaluation on the real CountyLevelEnv.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingToBlockTranslator:
    """Map embedding-space region+scenario decisions to CountyLevelEnv block IDs.

    Each block is assigned to its nearest embedding-space region based on
    cosine similarity between block GeoFM embeddings and region centers.
    """

    def __init__(self, block_embeddings, region_centers, block_features=None):
        """
        Args:
            block_embeddings: (n_blocks, 64) — per-block GeoFM embeddings
            region_centers: (n_regions, 64) — K-means cluster centers from EmbeddingSpaceEnv
            block_features: optional (n_blocks, 17) — block features from CountyLevelEnv
                            for scenario-aware block selection
        """
        self.block_emb = block_embeddings  # (2600, 64)
        self.region_centers = region_centers  # (K, 64)
        self.block_features = block_features
        self.n_blocks = block_embeddings.shape[0]
        self.n_regions = region_centers.shape[0]

        # Assign each block to nearest region
        sims = cosine_similarity(block_embeddings, region_centers)  # (2600, K)
        self.block_to_region = sims.argmax(axis=1)  # (2600,)

        # Build reverse map: region -> list of block indices
        self.region_to_blocks = {}
        for b in range(self.n_blocks):
            r = self.block_to_region[b]
            if r not in self.region_to_blocks:
                self.region_to_blocks[r] = []
            self.region_to_blocks[r].append(b)

    def translate(self, region_id, scenario_id=None):
        """Select the best block in a region for a given scenario.

        Args:
            region_id: embedding-space region index
            scenario_id: 0=urban, 1=eco_restore, 2=ag_intensify, 3=climate, 4=baseline

        Returns:
            block_id: int — best block index for CountyLevelEnv
        """
        blocks = self.region_to_blocks.get(region_id, [])
        if not blocks:
            return 0  # fallback

        if self.block_features is None or scenario_id is None:
            # No features — pick block with highest embedding norm (most "active")
            norms = np.linalg.norm(self.block_emb[blocks], axis=1)
            return blocks[norms.argmax()]

        bf = self.block_features[blocks]  # (n, 17)

        if scenario_id == 2:  # agricultural_intensification
            # Select block with highest slope gap (feature 3: best_gain)
            # Higher slope gap = more room for farmland improvement
            scores = bf[:, 3]
        elif scenario_id == 1:  # ecological_restoration
            # Select block with most forest area (feature 8: forest_area_norm)
            scores = bf[:, 8]
        else:
            # Default: highest swap potential (feature 9)
            scores = bf[:, 9]

        return blocks[scores.argmax()]

    def translate_episode(self, actions):
        """Translate a sequence of (region_id, scenario_id) to block_ids.

        Args:
            actions: list of (region_id, scenario_id) tuples

        Returns:
            block_ids: list of int
        """
        return [self.translate(r, s) for r, s in actions]

    def get_region_stats(self):
        """Summary statistics of block-to-region mapping."""
        sizes = [len(self.region_to_blocks.get(r, [])) for r in range(self.n_regions)]
        return {
            'n_regions': self.n_regions,
            'n_blocks': self.n_blocks,
            'blocks_per_region_mean': np.mean(sizes),
            'blocks_per_region_std': np.std(sizes),
            'blocks_per_region_min': min(sizes),
            'blocks_per_region_max': max(sizes),
            'empty_regions': sum(1 for s in sizes if s == 0),
        }


if __name__ == '__main__':
    import os, json

    # Load block embeddings
    block_emb = np.load('D:/test/paper7/data/block_geofm_embeddings.npy')
    print(f'Block embeddings: {block_emb.shape}')

    # Load region centers from a quick EmbeddingSpaceEnv partition
    from embedding_space_env import partition_regions
    emb_2020 = np.load(os.path.join(os.path.dirname(__file__), 'data', 'bishan_emb_2020.npy'))
    region_map, region_centers = partition_regions(emb_2020, n_regions=50)
    print(f'Region centers: {region_centers.shape}')

    translator = EmbeddingToBlockTranslator(block_emb, region_centers)
    stats = translator.get_region_stats()
    print(f'Mapping stats: {json.dumps(stats, indent=2)}')

    # Test translation
    for scenario in range(5):
        block = translator.translate(region_id=0, scenario_id=scenario)
        print(f'  Region 0, scenario {scenario} -> block {block}')
