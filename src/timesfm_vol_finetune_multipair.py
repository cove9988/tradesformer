"""Fine-tune TimesFM 2.5 (LoRA) on POOLED multi-pair forward volatility.

TEST 2 — generalization study
------------------------------
TEST 1 fine-tuned on EURUSD-only pre-2025 data and beat zero-shot on the EURUSD
2025-26 forward set. The open question: does pooling MORE data across MULTIPLE
currency pairs (a) help EURUSD further, and (b) generalize to pairs the adapter
was *also* trained on but that differ structurally (JPY cross, commodity CAD,
safe-haven CHF)?

This script pools the trailing-realized-vol series (the exact quantity
timesfm_vol_eval scores) across all requested pairs' pre-2025 train weeks, then
fine-tunes ONE shared LoRA adapter. Everything else (LoRA r/alpha/dropout,
context/horizon/window, optimizer, epochs) is held identical to the EURUSD-only
TEST-1 recipe so the ONLY changed variable is the training data.

Leakage control (the critical part)
-----------------------------------
FX volatility is strongly cross-pair correlated, so a naive random val split
would leak. We hold out, PER PAIR, the chronologically LAST `val_weeks_per_pair`
weeks of the pre-2025 train range for early-stopping only. The 2025-26 forward
test sets (data/split/forward/<PAIR>/weekly/test) are NEVER read here. Because
every pair's training data ends 2024-11 and the forward sets start 2025_08,
no forward week can enter training or validation.

Run
---
  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    python -m src.timesfm_vol_finetune_multipair \
      --pairs EURUSD GBPUSD USDCAD USDCHF USDJPY \
      --context-len 512 --horizon-len 96 --window 48 \
      --num-samples 8000 --epochs 8 --batch-size 16 --lr 5e-4 \
      --val-weeks-per-pair 21 \
      --output-dir data/adapters/multipair_vol_h96

  PYTHONPATH=. python -m src.timesfm_vol_finetune_multipair --backend fake  # dry run
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.timesfm_vol_eval import _trailing_rv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("timesfm_vol_finetune_multipair")

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_SPLIT_ROOT = "data/split"
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]


def list_pair_weeks(pair: str, split_root: str | Path,
                    subdir: str = "weekly/train") -> list[str]:
    """Return sorted (chronological) list of weekly CSV paths for a pair.

    Raises FileNotFoundError if the pair's train directory is missing/empty.
    """
    d = Path(split_root) / pair / subdir
    files = sorted(glob.glob(str(d / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No weekly train CSVs for {pair} at {d}")
    return files


def chrono_train_val_split(files: list[str], val_weeks: int) -> tuple[list[str], list[str]]:
    """Split a chronologically-sorted file list into (train, val).

    The LAST `val_weeks` entries become validation (early-stopping); everything
    before is training. This is strictly causal: val is always later in time
    than train for that pair.
    """
    if val_weeks <= 0 or val_weeks >= len(files):
        raise ValueError(
            f"val_weeks must be in 1..len(files)-1, got {val_weeks} for {len(files)} files")
    train = files[:-val_weeks]
    val = files[-val_weeks:]
    return train, val


def _vol_series_from_files(files: list[str], window: int,
                           target_col: str = "log_return_1") -> list[np.ndarray]:
    series = []
    for f in files:
        df = pd.read_csv(f).dropna(subset=[target_col]).reset_index(drop=True)
        r = df[target_col].to_numpy(dtype=np.float64)
        rv = _trailing_rv(r, window)
        rv = rv[~np.isnan(rv)].astype(np.float32)
        if len(rv) > 0:
            series.append(rv)
    return series


def load_multipair_vol_series(pairs, split_root, window: int,
                              val_weeks_per_pair: int):
    """Pool trailing-rv series across pairs with a per-pair chronological val holdout.

    Returns (train_series, val_series, meta) where train/val are flat lists of
    float32 rv arrays pooled across all pairs, and meta[pair] records the
    train/val week counts and chronological ranges (for provenance + leakage audit).
    """
    train_series: list[np.ndarray] = []
    val_series: list[np.ndarray] = []
    meta: dict[str, dict] = {}
    for pair in pairs:
        files = list_pair_weeks(pair, split_root)
        tr_files, va_files = chrono_train_val_split(files, val_weeks_per_pair)
        tr = _vol_series_from_files(tr_files, window)
        va = _vol_series_from_files(va_files, window)
        train_series.extend(tr)
        val_series.extend(va)
        meta[pair] = {
            "n_train": len(tr_files),
            "n_val": len(va_files),
            "train_range": [Path(tr_files[0]).stem, Path(tr_files[-1]).stem],
            "val_range": [Path(va_files[0]).stem, Path(va_files[-1]).stem],
        }
        logger.info("%s: train weeks=%d (%s..%s)  val weeks=%d (%s..%s)",
                    pair, len(tr_files), meta[pair]["train_range"][0], meta[pair]["train_range"][1],
                    len(va_files), meta[pair]["val_range"][0], meta[pair]["val_range"][1])
    return train_series, val_series, meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LoRA fine-tune TimesFM 2.5 on POOLED multi-pair volatility")
    p.add_argument("--model-id", dest="model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--split-root", dest="split_root", default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--context-len", dest="context_len", type=int, default=512)
    p.add_argument("--horizon-len", dest="horizon_len", type=int, default=96)
    p.add_argument("--val-weeks-per-pair", dest="val_weeks_per_pair", type=int, default=21)
    p.add_argument("--num-samples", dest="num_samples", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", dest="output_dir", default="data/adapters/multipair_vol_h96")
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

    train_series, val_series, meta = load_multipair_vol_series(
        a.pairs, a.split_root, a.window, a.val_weeks_per_pair)
    n_train_weeks = sum(m["n_train"] for m in meta.values())
    n_val_weeks = sum(m["n_val"] for m in meta.values())
    logger.info("POOLED %d pairs: train weeks=%d  val weeks=%d  (window=%d)",
                len(a.pairs), n_train_weeks, n_val_weeks, a.window)
    if not train_series or not val_series:
        raise SystemExit("No pooled vol series -- check pairs/split-root.")

    train_ds, _ = build_datasets(train_series, a.context_len, a.horizon_len,
                                 num_samples=a.num_samples, seed=a.seed)
    _, val_ds = build_datasets(val_series, a.context_len, a.horizon_len,
                               num_samples=1, seed=a.seed)
    logger.info("datasets: train_windows=%d  val_windows=%d", len(train_ds), len(val_ds))

    if a.backend == "fake":
        ctx, tgt = train_ds[0]
        logger.info("DRY RUN ok. pooled pairs=%s sample ctx=%s tgt=%s",
                    a.pairs, np.asarray(ctx).shape, np.asarray(tgt).shape)
        for pair, m in meta.items():
            logger.info("  %s train %s..%s | val %s..%s", pair,
                        m["train_range"][0], m["train_range"][1],
                        m["val_range"][0], m["val_range"][1])
        return

    import json
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

    # Provenance sidecar: exactly what data trained this adapter (leakage audit trail).
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    prov = {
        "experiment": "TEST2_multipair_vol",
        "pairs": a.pairs,
        "context_len": context_len, "horizon_len": a.horizon_len, "window": a.window,
        "num_samples": a.num_samples, "epochs": a.epochs, "batch_size": a.batch_size,
        "lr": a.lr, "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout},
        "val_weeks_per_pair": a.val_weeks_per_pair,
        "best_val_loss": best,
        "per_pair": meta,
        "note": "train ends 2024-11; forward test starts 2025_08 -> no leakage",
    }
    (Path(a.output_dir) / "provenance.json").write_text(json.dumps(prov, indent=2))
    logger.info("Wrote provenance -> %s/provenance.json", a.output_dir)


if __name__ == "__main__":
    main()
