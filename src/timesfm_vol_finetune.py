"""Fine-tune TimesFM 2.5 (LoRA) to forecast EURUSD *volatility*.

Why this script exists
----------------------
Zero-shot TimesFM already forecasts forward realized volatility better than the
standard baselines (see timesfm_vol_eval.py). Direction is ~unpredictable, so
volatility is the only place fine-tuning can plausibly add value. This script
fine-tunes on EXACTLY the series the eval scores:

    input/target series = trailing realized vol   rv[t] = sqrt(mean(r[t-W:t]^2))

The eval feeds TimesFM the rv series and forecasts it forward, then takes the RMS
of the predicted path as the forward-vol prediction. So we train the model on the
rv trajectory itself -- not raw returns. Training on raw returns would optimise a
different quantity than we measure (the direction-null target), which is why the
existing timesfm25_finetune.py target is wrong for this purpose.

Leakage control
---------------
Series are filtered by the split manifest (data/split/EURUSD/weekly/manifest.json):
    train weeks -> random-window training set
    val   weeks -> last-window early-stopping set
TEST weeks are NEVER touched here; they are held out for timesfm_vol_eval --split test.

Run
---
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    python -m src.timesfm_vol_finetune \
      --context-len 512 --horizon-len 48 --window 48 \
      --num-samples 4000 --epochs 8 --batch-size 16 \
      --output-dir data/adapters/eurusd_vol_h48

  PYTHONPATH=. python -m src.timesfm_vol_finetune --backend fake   # data-pipeline dry run
"""
from __future__ import annotations

import argparse
import glob
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_split import load_split_manifest
from src.timesfm_vol_eval import _trailing_rv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("timesfm_vol_finetune")

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_DATA_DIR = "data/split/EURUSD/weekly/train"
DEFAULT_MANIFEST = "data/split/EURUSD/weekly/manifest.json"


def load_vol_series(data_dir, manifest_path, split, window, target_col="log_return_1"):
    """Return a list of trailing-realized-vol arrays, one per week in `split`.

    Each array is the rv[t] = sqrt(mean(r[t-window:t]^2)) trajectory for that
    week, with leading NaNs (before `window` bars of history) dropped. This is
    exactly the series timesfm_vol_eval feeds the model.
    """
    weeks = None
    if split and split != "all":
        m = load_split_manifest(manifest_path)
        weeks = set(m.get("splits", {}).get(split, []))
        if not weeks:
            raise ValueError(f"split '{split}' empty/missing in {manifest_path}")
    files = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if weeks is not None:
        files = [f for f in files if Path(f).name in weeks]
    series = []
    for f in files:
        df = pd.read_csv(f).dropna(subset=[target_col]).reset_index(drop=True)
        r = df[target_col].to_numpy(dtype=np.float64)
        rv = _trailing_rv(r, window)
        rv = rv[~np.isnan(rv)].astype(np.float32)
        if len(rv) > 0:
            series.append(rv)
    return series


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LoRA fine-tune TimesFM 2.5 on EURUSD volatility")
    p.add_argument("--model-id", dest="model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--manifest-path", dest="manifest_path", default=DEFAULT_MANIFEST)
    p.add_argument("--window", type=int, default=48, help="trailing-vol window (bars)")
    p.add_argument("--context-len", dest="context_len", type=int, default=512,
                   help="must be a multiple of 32")
    p.add_argument("--horizon-len", dest="horizon_len", type=int, default=48)
    p.add_argument("--num-samples", dest="num_samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", dest="output_dir", default="data/adapters/eurusd_vol")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    p.add_argument("--backend", default="timesfm-2.5-transformers",
                   choices=["timesfm-2.5-transformers", "fake"])
    return p.parse_args(argv)


def main(argv=None):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    a = parse_args(argv)
    if a.context_len % 32 != 0:
        raise ValueError(f"--context-len must be multiple of 32, got {a.context_len}")

    from src.timesfm25_finetune import build_datasets, make_lora_config, run_training

    train_series = load_vol_series(a.data_dir, a.manifest_path, "train", a.window)
    val_series = load_vol_series(a.data_dir, a.manifest_path, "val", a.window)
    logger.info("vol series: train weeks=%d  val weeks=%d  (window=%d)",
                len(train_series), len(val_series), a.window)
    if not train_series or not val_series:
        raise SystemExit("No vol series for train/val -- check manifest/splits.")

    # Train on random windows of TRAIN-week rv series; early-stop on VAL-week rv.
    train_ds, _ = build_datasets(train_series, a.context_len, a.horizon_len,
                                 num_samples=a.num_samples, seed=a.seed)
    _, val_ds = build_datasets(val_series, a.context_len, a.horizon_len,
                               num_samples=1, seed=a.seed)
    logger.info("datasets: train_windows=%d  val_windows=%d", len(train_ds), len(val_ds))

    if a.backend == "fake":
        ctx, tgt = train_ds[0]
        logger.info("DRY RUN ok. sample ctx=%s tgt=%s", np.asarray(ctx).shape, np.asarray(tgt).shape)
        return

    import torch
    from peft import get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    logger.info("device=%s dtype=%s", device, torch_dtype)

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        a.model_id, torch_dtype=torch_dtype, device_map=device)
    context_len = min(a.context_len, model.config.context_length)
    model = get_peft_model(model, make_lora_config(a.lora_r, a.lora_alpha, a.lora_dropout))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs * max(len(train_ds) // a.batch_size, 1))

    best = run_training(model, train_ds, val_ds, optimizer, scheduler,
                        context_len, a.horizon_len, a.epochs, a.batch_size,
                        a.output_dir, device)
    logger.info("Done. best_val_loss=%.6f  adapter=%s", best, a.output_dir)


if __name__ == "__main__":
    main()
