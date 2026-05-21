"""Fine-tune TimesFM 2.5 with LoRA/DoRA on your MT5 financial data.

Uses the PEFT pipeline from peft/ to fine-tune the model on all
configured instruments simultaneously.

Run:
    python mt5/scripts/finetune.py
    python mt5/scripts/finetune.py --adapter_type dora --num_epochs 30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def detect_hardware():
    """Detect and report all available compute devices."""
    print("\n--- Hardware Detection ---")

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  CUDA:{i}  {props.name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("  CUDA:    Not available")

    # Intel XPU (Arc iGPU) — via intel_extension_for_pytorch
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                name = torch.xpu.get_device_name(i)
                print(f"  XPU:{i}   {name}  (shared RAM — not used for training)")
        else:
            print("  XPU:     Not available")
    except ImportError:
        print("  XPU:     intel_extension_for_pytorch not installed (Intel iGPU skipped)")

    # Intel NPU — via openvino
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        npu_devices = [d for d in devices if "NPU" in d]
        if npu_devices:
            for d in npu_devices:
                print(f"  NPU:     {d} detected (inference only — not used for training)")
        else:
            print("  NPU:     Not detected")
    except ImportError:
        print("  NPU:     openvino not installed (Intel NPU skipped)")

    # Select best training device
    if torch.cuda.is_available():
        # Pick the NVIDIA GPU (highest VRAM)
        best_idx = 0
        best_vram = 0
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory
            if vram > best_vram:
                best_vram = vram
                best_idx = i
        device = torch.device(f"cuda:{best_idx}")
        name = torch.cuda.get_device_properties(best_idx).name
        print(f"\n  >> Training device: {device}  ({name})")
    else:
        device = torch.device("cpu")
        print(f"\n  >> Training device: CPU (no CUDA GPU found)")

    print("---\n")
    return device

from config import (
    ADAPTER_TYPE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CONTEXT_LEN,
    COV_DIR,
    EARLY_STOPPING_PATIENCE,
    HORIZON_LEN,
    INSTRUMENTS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    NUM_ADAPTER_LAYERS,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    TIMEFRAME,
)

import timesfm
from peft import PEFTConfig, PEFTTrainer, TimeSeriesDataset


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TimesFM 2.5 on MT5 data")
    p.add_argument("--adapter_type", default=ADAPTER_TYPE, choices=["lora", "dora"])
    p.add_argument("--lora_rank", type=int, default=LORA_RANK)
    p.add_argument("--lora_alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--num_adapter_layers", type=int, default=NUM_ADAPTER_LAYERS)
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    p.add_argument("--horizon_len", type=int, default=HORIZON_LEN)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def load_all_series() -> list[np.ndarray]:
    """Load close prices from all instruments' covariate CSVs."""
    tf_label = TIMEFRAME.replace("TIMEFRAME_", "")
    all_series = []

    for name in INSTRUMENTS:
        path = COV_DIR / f"{name}_{tf_label}_covariates.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue

        df = pd.read_csv(path)
        closes = df["close"].dropna().values.astype(np.float32)
        if len(closes) > 0:
            all_series.append(closes)
            print(f"  {name}: {len(closes):,} bars")

    return all_series


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    print("=" * 60)
    print("TimesFM 2.5 PEFT Fine-Tuning — MT5 Financial Data")
    print("=" * 60)

    # Detect hardware and select training device
    device = detect_hardware()

    # Load data
    print("\nLoading time series data...")
    all_series = load_all_series()
    if not all_series:
        print("No data found. Run export_mt5_data.py and build_covariates.py first.")
        sys.exit(1)

    print(f"\nTotal instruments: {len(all_series)}")
    total_bars = sum(len(s) for s in all_series)
    print(f"Total bars: {total_bars:,}")

    # Split: last 20% of each series for validation (chronological split)
    train_series = []
    val_series = []
    for s in all_series:
        split_idx = int(len(s) * 0.8)
        train_series.append(s[:split_idx])
        val_series.append(s[split_idx:])

    # Create datasets
    print(f"\nCreating datasets (context={args.context_len}, horizon={args.horizon_len})...")
    stride = args.context_len // 2  # 50% overlap for training diversity
    train_ds = TimeSeriesDataset(
        train_series,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        stride=stride,
    )
    val_ds = TimeSeriesDataset(
        val_series,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        stride=stride,
    )
    print(f"Train windows: {len(train_ds):,}")
    print(f"Val windows:   {len(val_ds):,}")

    # Load model
    print("\nLoading TimesFM 2.5...")
    torch.set_float32_matmul_precision("high")
    # Pin to the selected CUDA device before loading
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,  # Must disable for fine-tuning
    )

    # Configure PEFT
    config = PEFTConfig(
        adapter_type=args.adapter_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_adapter_layers=args.num_adapter_layers,
        train_output_head=False,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_clip_norm=1.0,
        warmup_ratio=WARMUP_RATIO,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb,
        wandb_project="timesfm-mt5-financial",
        checkpoint_dir=str(CHECKPOINT_DIR),
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    print(f"\nPEFT config:")
    print(f"  Adapter: {config.adapter_type} (rank={config.lora_rank}, alpha={config.lora_alpha}, dropout={config.lora_dropout})")
    print(f"  Layers adapted: last {config.num_adapter_layers} of 20")
    print(f"  LR: {config.learning_rate}, Epochs: {config.num_epochs}")
    print(f"  Weight decay: {config.weight_decay}, Warmup: {config.warmup_ratio}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Mixed precision: {config.mixed_precision}")

    # Train
    print("\nStarting fine-tuning...")
    trainer = PEFTTrainer(model_wrapper.model, config)
    history = trainer.fit(train_ds, val_ds)

    # Save
    adapter_path = str(CHECKPOINT_DIR / "mt5_financial_adapter.safetensors")
    trainer.save_adapter(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")

    # Summary
    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if history.get("val_loss"):
        print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
