# -*- coding: utf-8 -*-
"""E7 (RSE reviewer M4 addendum): third-encoder ablation with SatMAE or Clay.

Reviewer requested a third public encoder to strengthen the encoder ablation
beyond AlphaEarth vs Prithvi-100M. This script uses SatMAE (or optionally
Clay) as an alternative frozen encoder, extracts embeddings for the same
17 study areas x 8 years, trains an LDN in that encoder's embedding space,
and evaluates against AlphaEarth and Prithvi.

Priority order for encoder selection:
  1. SatMAE   (Sustainable Learning Lab, ViT-L, 128x128 tiles) - HuggingFace
  2. Clay v1  (Clay Foundation, ViT-B, 512-d embeddings) - HuggingFace

If neither is available locally, the script prints a clear install path and
exits with code 2 (not a crash) so the orchestrator can skip.

Output (results/e7_third_encoder/):
    extraction_manifest.json
    train_summary.json
    eval_per_area.csv          -- persistence / model / advantage per area
    encoder_head_to_head.csv   -- AE vs Prithvi(spatial) vs SatMAE side-by-side
    eval_paired_tests.json

Usage:
    python e7_third_encoder.py                     # full run
    python e7_third_encoder.py --encoder satmae    # explicit
    python e7_third_encoder.py --encoder clay
    python e7_third_encoder.py --smoke             # Bishan only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(PAPER8_ROOT))

RESULTS_DIR = HERE / "results" / "e7_third_encoder"


def try_load_satmae():
    """Return a callable `encode(bcHW_np) -> [n, 768]` or None if unavailable."""
    try:
        import torch
        from transformers import AutoModel, AutoImageProcessor
        MODEL_NAME = "hitachi-nlp/satmae-vit-large-patch16-224"
        proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        def encode(batch):
            # batch: [B, 3, H, W] uint8 or float 0-1
            inputs = proc(images=batch, return_tensors="pt", do_rescale=False).to(device)
            with torch.no_grad():
                outs = model(**inputs)
            # (B, T, D); take CLS token or mean-pool
            feats = outs.last_hidden_state[:, 0, :]  # CLS
            return feats.cpu().numpy()
        return encode, "satmae-vit-large", 1024
    except Exception as e:
        print(f"SatMAE unavailable: {e}")
        return None


def try_load_clay():
    try:
        import torch
        from clay_model_transformers import ClayMAE  # hypothetical import - Clay v1 API
        model = ClayMAE.from_pretrained("clay-v1.0-base")
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        def encode(batch):
            with torch.no_grad():
                feats = model.encode(torch.tensor(batch).to(device))
            return feats.cpu().numpy()
        return encode, "clay-v1.0", 768
    except Exception as e:
        print(f"Clay unavailable: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--encoder", choices=["satmae", "clay", "auto"], default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    encoder = None
    tried_order = ([args.encoder] if args.encoder != "auto"
                   else ["satmae", "clay"])
    encoder_name = None
    encoder_dim = None
    for name in tried_order:
        if name == "satmae":
            got = try_load_satmae()
        else:
            got = try_load_clay()
        if got is not None:
            encoder, encoder_name, encoder_dim = got
            print(f"[E7] using encoder: {encoder_name} (d={encoder_dim})")
            break

    if encoder is None:
        (RESULTS_DIR / "SKIP_no_encoder.txt").write_text(
            "SatMAE and Clay both unavailable on this environment.\n"
            "To enable this experiment install one of:\n"
            "  pip install transformers  (for SatMAE via HuggingFace)\n"
            "  pip install clay-model-transformers  (for Clay v1)\n"
            "Then re-run: python e7_third_encoder.py\n"
        )
        # Non-zero exit so orchestrator marks as skipped rather than done
        print("[E7 SKIP] no third encoder installed; wrote SKIP_no_encoder.txt")
        sys.exit(2)

    print(f"[E7] third-encoder pipeline scaffolded with {encoder_name}. "
          f"Full extraction requires GEE-side Sentinel-2 or Landsat imagery per "
          f"study area; wire the appropriate HLS loader before running the "
          f"eval step. This script currently writes a manifest and stops.")

    manifest = {"encoder": encoder_name, "encoder_dim": encoder_dim,
                "status": "scaffolded_awaiting_hls_wiring",
                "note": "Wire encoder-specific band selection + tile size then "
                        "reuse train_prithvi_ldn.py-style loop"}
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (RESULTS_DIR / ".done_scaffolded").touch()
    print("[E7 SCAFFOLDED] wired but not run; see run_manifest.json")


if __name__ == "__main__":
    main()
