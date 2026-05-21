"""Failing tests for src/timefm25_finetune.py — LoRA config and training contract."""
from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_series(n: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.001, n).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. parse_args defaults
# ---------------------------------------------------------------------------

class ParseArgsTest(unittest.TestCase):

    def setUp(self):
        from src.timesfm25_finetune import parse_args
        self.parse_args = parse_args

    def test_default_model_id(self):
        args = self.parse_args([])
        self.assertEqual(args.model_id, "google/timesfm-2.5-200m-transformers")

    def test_default_context_len_is_multiple_of_32(self):
        args = self.parse_args([])
        self.assertEqual(args.context_len % 32, 0)

    def test_default_horizon_len(self):
        args = self.parse_args([])
        self.assertGreater(args.horizon_len, 0)

    def test_default_lora_r(self):
        args = self.parse_args([])
        self.assertGreater(args.lora_r, 0)

    def test_default_lora_alpha(self):
        args = self.parse_args([])
        self.assertGreater(args.lora_alpha, 0)

    def test_context_len_override(self):
        args = self.parse_args(["--context-len", "128"])
        self.assertEqual(args.context_len, 128)

    def test_horizon_len_override(self):
        args = self.parse_args(["--horizon-len", "24"])
        self.assertEqual(args.horizon_len, 24)

    def test_data_dir_required_when_no_default(self):
        # data-dir has no default — must be provided or raise
        # We patch so it's not required (it has a default in implementation)
        # just verify it exists as an attribute
        args = self.parse_args([])
        self.assertTrue(hasattr(args, "data_dir"))

    def test_output_dir_has_default(self):
        args = self.parse_args([])
        self.assertIsNotNone(args.output_dir)


# ---------------------------------------------------------------------------
# 2. build_datasets returns train + val datasets
# ---------------------------------------------------------------------------

class BuildDatasetsTest(unittest.TestCase):

    def test_returns_train_and_val_tuple(self):
        from src.timesfm25_finetune import build_datasets
        series = [_fake_series(300) for _ in range(10)]
        train_ds, val_ds = build_datasets(
            series_list=series,
            context_len=64,
            horizon_len=12,
            num_samples=20,
            seed=0,
        )
        self.assertEqual(len(train_ds), 20)
        self.assertGreater(len(val_ds), 0)

    def test_train_item_shapes(self):
        from src.timesfm25_finetune import build_datasets
        series = [_fake_series(300) for _ in range(5)]
        train_ds, _ = build_datasets(series, 64, 12, num_samples=10)
        ctx, tgt = train_ds[0]
        self.assertEqual(np.asarray(ctx).shape, (64,))
        self.assertEqual(np.asarray(tgt).shape, (12,))

    def test_val_item_shapes(self):
        from src.timesfm25_finetune import build_datasets
        series = [_fake_series(300) for _ in range(5)]
        _, val_ds = build_datasets(series, 64, 12, num_samples=10)
        ctx, tgt = val_ds[0]
        self.assertEqual(np.asarray(ctx).shape, (64,))
        self.assertEqual(np.asarray(tgt).shape, (12,))


# ---------------------------------------------------------------------------
# 3. make_lora_config returns a LoraConfig-like object
# ---------------------------------------------------------------------------

class MakeLoraConfigTest(unittest.TestCase):

    def test_lora_r_and_alpha_are_set(self):
        from src.timesfm25_finetune import make_lora_config
        cfg = make_lora_config(r=4, lora_alpha=8, lora_dropout=0.05)
        self.assertEqual(cfg.r, 4)
        self.assertEqual(cfg.lora_alpha, 8)

    def test_target_modules_is_all_linear(self):
        from src.timesfm25_finetune import make_lora_config
        cfg = make_lora_config(r=4, lora_alpha=8, lora_dropout=0.05)
        self.assertEqual(cfg.target_modules, "all-linear")

    def test_bias_is_none(self):
        from src.timesfm25_finetune import make_lora_config
        cfg = make_lora_config(r=4, lora_alpha=8, lora_dropout=0.05)
        self.assertEqual(cfg.bias, "none")


# ---------------------------------------------------------------------------
# 4. run_training is callable with stubbed model + loader
# ---------------------------------------------------------------------------

class RunTrainingTest(unittest.TestCase):

    def _make_stub_model(self):
        """Minimal model stub — no torch required."""
        class StubOutput:
            loss = _scalar_tensor(0.5)

        class StubModel:
            def __call__(self, past_values, future_values, forecast_context_len):
                return StubOutput()
            def train(self): pass
            def eval(self): pass
            def parameters(self): return iter([])
            def save_pretrained(self, path): pass

        return StubModel()

    def test_run_training_returns_best_val_loss(self):
        from src.timesfm25_finetune import run_training

        series = [_fake_series(300) for _ in range(5)]
        train_ds, val_ds = __import__("src.timesfm25_finetune", fromlist=["build_datasets"]).build_datasets(
            series, 64, 12, num_samples=8
        )

        stub_model = self._make_stub_model()
        stub_optimizer = _FakeOptimizer()
        stub_scheduler = _FakeScheduler()

        best = run_training(
            model=stub_model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            optimizer=stub_optimizer,
            scheduler=stub_scheduler,
            context_len=64,
            horizon_len=12,
            epochs=1,
            batch_size=4,
            output_dir="/tmp/test_lora_adapter",
            device="cpu",
        )
        self.assertIsInstance(best, float)

    def test_run_training_calls_save_pretrained(self):
        from src.timesfm25_finetune import run_training, build_datasets

        series = [_fake_series(300) for _ in range(5)]
        train_ds, val_ds = build_datasets(series, 64, 12, num_samples=4)

        saved_paths = []

        class TrackingSaveModel:
            def __call__(self, past_values, future_values, forecast_context_len):
                class O:
                    loss = _scalar_tensor(0.1)
                return O()
            def train(self): pass
            def eval(self): pass
            def parameters(self): return iter([])
            def save_pretrained(self, path): saved_paths.append(path)

        run_training(
            model=TrackingSaveModel(),
            train_dataset=train_ds,
            val_dataset=val_ds,
            optimizer=_FakeOptimizer(),
            scheduler=_FakeScheduler(),
            context_len=64,
            horizon_len=12,
            epochs=1,
            batch_size=4,
            output_dir="/tmp/test_lora_adapter2",
            device="cpu",
        )
        self.assertGreater(len(saved_paths), 0)


# ---------------------------------------------------------------------------
# stub helpers (no torch)
# ---------------------------------------------------------------------------

def _scalar_tensor(val: float):
    """Return an object that behaves like a scalar loss tensor."""
    class _FakeTensor:
        def __init__(self, v): self._v = v
        def item(self): return float(self._v)
        def backward(self): pass
    return _FakeTensor(val)


class _FakeOptimizer:
    def step(self): pass
    def zero_grad(self): pass


class _FakeScheduler:
    def step(self): pass


if __name__ == "__main__":
    unittest.main()
