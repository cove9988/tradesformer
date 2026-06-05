"""TimesFM 2.5 LoRA fine-tuning script.

Local development and testing:
    python src/timesfm25_finetune.py --data-dir data/split/EURUSD/weekly/train \
        --context-len 64 --horizon-len 12 --epochs 2 --batch-size 8 \
        --backend fake --output-dir /tmp/test_lora

Colab / GPU training (push repo to GitHub, clone in Colab):
    python src/timesfm25_finetune.py \
        --data-dir data/split/EURUSD/weekly/train \
        --context-len 64 --horizon-len 12 \
        --epochs 10 --batch-size 32 \
        --output-dir /content/drive/MyDrive/tradesformer-lora

Public API (importable for tests):
    parse_args(argv)
    build_datasets(series_list, context_len, horizon_len, num_samples, seed)
    make_lora_config(r, lora_alpha, lora_dropout)
    run_training(model, train_dataset, val_dataset, optimizer, scheduler,
                 context_len, horizon_len, epochs, batch_size, output_dir, device)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.timesfm25_torch_dataset import (
    TimeSeriesLastWindowDataset,
    TimeSeriesRandomWindowDataset,
    load_series_from_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_CONTEXT_LEN = 64        # multiple of 32 (TimesFM patch size)
DEFAULT_HORIZON_LEN = 12
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_LORA_R = 4
DEFAULT_LORA_ALPHA = 8
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_OUTPUT_DIR = "tradesformer-lora"
DEFAULT_TARGET_COL = "log_return_1"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Public helpers (importable, torch-free)
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pass argv=[] for defaults in tests."""
    p = argparse.ArgumentParser(
        description="Fine-tune TimesFM 2.5 with LoRA on EURUSD weekly CSV data"
    )
    p.add_argument("--model-id", dest="model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-dir", dest="data_dir",
                   default="data/split/EURUSD/weekly/train",
                   help="Directory of weekly EURUSD CSV files")
    p.add_argument("--target-col", dest="target_col", default=DEFAULT_TARGET_COL)
    p.add_argument("--context-len", dest="context_len",
                   type=int, default=DEFAULT_CONTEXT_LEN,
                   help="Context length (must be multiple of 32)")
    p.add_argument("--horizon-len", dest="horizon_len",
                   type=int, default=DEFAULT_HORIZON_LEN)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", dest="batch_size",
                   type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lora-r", dest="lora_r",
                   type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", dest="lora_alpha",
                   type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", dest="lora_dropout",
                   type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--num-samples", dest="num_samples",
                   type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Random training windows to pre-sample")
    p.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--backend", default="timesfm-2.5-transformers",
                   choices=["timesfm-2.5-transformers", "fake"],
                   help="'fake' runs a dry-run loop without loading the real model")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"],
                   help="auto = float32 on cpu/mps (stable), bfloat16 on cuda")
    return p.parse_args(argv)


def build_datasets(
    series_list: list[np.ndarray],
    context_len: int,
    horizon_len: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset]:
    """Build train (random windows) and val (last windows) datasets.

    Returns:
        (train_dataset, val_dataset)
    """
    train_ds = TimeSeriesRandomWindowDataset(
        series_list=series_list,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=num_samples,
        seed=seed,
    )
    val_ds = TimeSeriesLastWindowDataset(
        series_list=series_list,
        context_len=context_len,
        horizon_len=horizon_len,
    )
    return train_ds, val_ds


def make_lora_config(
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
) -> Any:
    """Build a PEFT LoraConfig targeting all linear layers.

    Returns a peft.LoraConfig instance.
    """
    from peft import LoraConfig  # type: ignore
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
    )


def run_training(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    optimizer: Any,
    scheduler: Any,
    context_len: int,
    horizon_len: int,
    epochs: int,
    batch_size: int,
    output_dir: str,
    device: str = "cpu",
) -> float:
    """Core training loop. Accepts any model with the TimesFM 2.5 call signature.

    The model must support:
        model(past_values=..., future_values=..., forecast_context_len=...) -> output
        output.loss.item() -> float
        output.loss.backward()
        model.save_pretrained(path)
        model.train() / model.eval()

    Returns:
        best_val_loss (float)
    """
    try:
        import torch
        from torch.utils.data import DataLoader

        def _to_tensor(arr):
            return torch.tensor(np.asarray(arr), dtype=torch.float32).to(device)

        def _collate(batch):
            ctxs = torch.stack([torch.tensor(c, dtype=torch.float32) for c, _ in batch])
            tgts = torch.stack([torch.tensor(t, dtype=torch.float32) for _, t in batch])
            return ctxs.to(device), tgts.to(device)

    except ImportError:
        # Torch not installed — running under test stubs.
        # Use a minimal list-based "loader" so run_training stays testable.
        class DataLoader:  # type: ignore
            def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=None):
                self._ds = dataset
                self._bs = batch_size
                self._shuffle = shuffle
                self._drop_last = drop_last
                self._collate_fn = collate_fn

            def __iter__(self):
                indices = list(range(len(self._ds)))
                for start in range(0, len(indices), self._bs):
                    batch_idx = indices[start : start + self._bs]
                    if self._drop_last and len(batch_idx) < self._bs:
                        continue
                    batch = [self._ds[i] for i in batch_idx]
                    ctxs = np.stack([np.asarray(c) for c, _ in batch])
                    tgts = np.stack([np.asarray(t) for _, t in batch])
                    yield ctxs, tgts

            def __len__(self):
                n = len(self._ds) // self._bs
                return n if self._drop_last else max(n, 1)

        def _collate(batch):
            return batch

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=_collate)

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for context, target_vals in train_loader:
            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            try:
                import torch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except (ImportError, AttributeError, StopIteration):
                pass
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        for context, target_vals in val_loader:
            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            val_loss += outputs.loss.item()
            val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        logger.info(
            "Epoch %d/%d — train_loss=%.5f val_loss=%.5f",
            epoch, epochs, avg_train, avg_val,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_dir)
            logger.info("  -> saved adapter to %s", output_dir)

    return best_val_loss


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate context_len is multiple of 32
    if args.context_len % 32 != 0:
        raise ValueError(
            f"--context-len must be a multiple of 32 (TimesFM patch size), "
            f"got {args.context_len}"
        )

    # Load data
    logger.info("Loading series from %s ...", args.data_dir)
    series_list = load_series_from_dir(
        data_dir=args.data_dir,
        target_col=args.target_col,
    )
    logger.info("Loaded %d series", len(series_list))

    # Build datasets
    context_len = args.context_len
    horizon_len = args.horizon_len
    train_ds, val_ds = build_datasets(
        series_list=series_list,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    if args.backend == "fake":
        # Dry-run: exercise the data pipeline without loading the model
        logger.info("Dry-run mode — skipping model load and training.")
        logger.info("Data pipeline OK. context_len=%d horizon_len=%d", context_len, horizon_len)
        ctx, tgt = train_ds[0]
        logger.info("Sample shapes: context=%s target=%s", np.asarray(ctx).shape, np.asarray(tgt).shape)
        return

    import torch
    from peft import get_peft_model  # type: ignore
    from transformers import TimesFm2_5ModelForPrediction  # type: ignore

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Resolve dtype: bfloat16 is well-supported on CUDA but flaky on MPS;
    # default to float32 on cpu/mps for numerical stability (200M fits easily).
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    logger.info("Dtype: %s", torch_dtype)

    # Load model
    logger.info("Loading model: %s", args.model_id)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    context_len = min(context_len, model.config.context_length)

    # Apply LoRA
    lora_cfg = make_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(train_ds) // args.batch_size, 1)
    )

    best = run_training(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        context_len=context_len,
        horizon_len=horizon_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=device,
    )
    logger.info("Done. Best val loss: %.5f | Adapter saved to: %s", best, args.output_dir)


if __name__ == "__main__":
    main()
