"""Failing tests for the TimesFM 2.5 Transformers inference contract.

These tests describe what src/timesfm_forecaster.py should look like AFTER the
v1 JAX/PAX path is retired.  They are written first and expected to FAIL until
the implementation is updated.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.timesfm_dataset import build_timesfm_samples
from src.timesfm_forecaster import (
    TimeFMForecastResult,
    TimeFMModelHandle,
    forecast_batch,
    forecast_sample,
    load_model,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sample(context_window: int = 4, forecast_horizon: int = 2):
    bars = context_window + forecast_horizon + 2
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=bars, freq="5min", tz="UTC"),
        "symbol": ["EURUSD"] * bars,
        "open": [1.1] * bars,
        "high": [1.11] * bars,
        "low": [1.09] * bars,
        "close": [1.1] * bars,
        "log_return_1": [0.001 * i for i in range(bars)],
        "close_rel_week_open": list(range(bars)),
        "hour_sin": [0.0] * bars,
    })
    return build_timesfm_samples(
        frame,
        context_window=context_window,
        forecast_horizon=forecast_horizon,
        target_col="log_return_1",
        feature_cols=["close_rel_week_open", "hour_sin"],
        source_file="EURUSD_2024_01.csv",
    )[0]


def _make_stub_transformers_module(mean_preds: np.ndarray):
    """Return a fake transformers module with TimesFm2_5ModelForPrediction.

    Deliberately torch-free so tests run before torch is installed.
    The stub output emulates the mean_predictions attribute as a numpy-backed
    object with the minimal interface the implementation needs.
    """

    class _TensorLike:
        """Minimal stand-in for a [batch, seq] shaped tensor."""
        def __init__(self, data: np.ndarray):
            self._data = np.asarray(data, dtype=np.float32)

        def __getitem__(self, item):
            result = self._data[item]
            if isinstance(result, np.ndarray):
                return _TensorLike(result)
            return float(result)

        @property
        def shape(self):
            return self._data.shape

        def numel(self):
            return self._data.size

        def cpu(self):
            return self

        def numpy(self):
            return self._data

        def float(self):
            return self

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

    class StubOutput:
        def __init__(self):
            # shape [1, len(mean_preds)]
            self.mean_predictions = _TensorLike(mean_preds[np.newaxis, :])
            self.loss = None

    class StubModel:
        def __init__(self, model_id, **kwargs):
            self._model_id = model_id
            self._calls = []

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            obj = cls(model_id)
            obj._from_pretrained_kwargs = kwargs
            return obj

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, past_values=None, future_values=None, forecast_context_len=None):
            self._calls.append({
                "past_values": past_values,
                "future_values": future_values,
                "forecast_context_len": forecast_context_len,
            })
            return StubOutput()

    class StubTransformers:
        TimesFm2_5ModelForPrediction = StubModel

    return StubTransformers, StubModel


# ---------------------------------------------------------------------------
# 1. load_model defaults
# ---------------------------------------------------------------------------

class LoadModelDefaultsTest(unittest.TestCase):
    """load_model() must default to the 2.5 Transformers model id and backend."""

    def test_default_model_id_is_timesfm25_transformers(self):
        stub_transformers, StubModel = _make_stub_transformers_module(
            np.zeros(12, dtype=np.float32)
        )
        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model()

        self.assertEqual(handle.model_id, "google/timesfm-2.5-200m-transformers")

    def test_default_backend_name_is_timesfm25_transformers(self):
        stub_transformers, _ = _make_stub_transformers_module(
            np.zeros(12, dtype=np.float32)
        )
        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model()

        self.assertEqual(handle.backend_name, "timesfm-2.5-transformers")

    def test_load_model_returns_timefm_model_handle(self):
        stub_transformers, _ = _make_stub_transformers_module(
            np.zeros(12, dtype=np.float32)
        )
        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model()

        self.assertIsInstance(handle, TimeFMModelHandle)

    def test_fake_backend_still_works_without_transformers(self):
        handle = load_model(backend_name="fake-timesfm")
        self.assertEqual(handle.backend_name, "fake-timesfm")
        self.assertEqual(handle.model_id, "google/timesfm-2.5-200m-transformers")


# ---------------------------------------------------------------------------
# 2. v1 path is gone
# ---------------------------------------------------------------------------

class V1PathRemovedTest(unittest.TestCase):
    """The v1 backend_name='timesfm' path must be removed."""

    def test_loading_v1_backend_raises_value_error(self):
        with self.assertRaises((ValueError, NotImplementedError)):
            load_model(model_id="google/timesfm-1.0-200m", backend_name="timesfm")


# ---------------------------------------------------------------------------
# 3. Inference call uses past_values tensor interface
# ---------------------------------------------------------------------------

class TransformersInferenceCallTest(unittest.TestCase):
    """The 2.5 Transformers path must call model(past_values=...) not model.forecast(...)."""

    def setUp(self):
        horizon = 2
        self._horizon = horizon
        context = 4
        self._sample = _make_sample(context_window=context, forecast_horizon=horizon)
        self._stub_transformers, self._StubModel = _make_stub_transformers_module(
            np.array([0.11, 0.22], dtype=np.float32)
        )

    def test_inference_uses_past_values_keyword(self):
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            forecast_sample(handle, self._sample)

        stub_model = handle.model
        self.assertTrue(len(stub_model._calls) > 0, "model was not called")
        last_call = stub_model._calls[-1]
        self.assertIn("past_values", last_call)
        self.assertIsNotNone(last_call["past_values"])

    def test_inference_does_not_use_forecast_method(self):
        """Regression: ensure old v1 forecast([context], freq=[0]) path is not used."""
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            result = forecast_sample(handle, self._sample)
        stub_model = handle.model
        self.assertFalse(
            hasattr(stub_model, "_forecast_called") and stub_model._forecast_called,
            "old .forecast() method should not be called on 2.5 backend"
        )

    def test_past_values_tensor_shape_matches_context_window(self):
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            forecast_sample(handle, self._sample)

        last_call = handle.model._calls[-1]
        pv = last_call["past_values"]
        # may be [1, context_window] or [context_window] — count total scalars
        arr = np.asarray(pv)
        self.assertEqual(arr.size, self._sample.context_window)


# ---------------------------------------------------------------------------
# 4. ForecastResult is populated from mean_predictions
# ---------------------------------------------------------------------------

class ForecastResultShapeTest(unittest.TestCase):

    def setUp(self):
        self._sample = _make_sample(context_window=4, forecast_horizon=2)
        self._horizon = 2
        stub_mean = np.array([0.111, 0.222], dtype=np.float32)
        self._expected_mean = stub_mean
        self._stub_transformers, _ = _make_stub_transformers_module(stub_mean)

    def test_predicted_mean_shape_equals_forecast_horizon(self):
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            result = forecast_sample(handle, self._sample)

        self.assertEqual(result.predicted_mean.shape, (self._horizon,))

    def test_predicted_mean_values_come_from_model_output(self):
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            result = forecast_sample(handle, self._sample)

        np.testing.assert_allclose(
            result.predicted_mean, self._expected_mean, rtol=1e-5
        )

    def test_result_carries_correct_metadata(self):
        with patch.dict(sys.modules, {"transformers": self._stub_transformers}):
            handle = load_model()
            result = forecast_sample(handle, self._sample)

        self.assertIsInstance(result, TimeFMForecastResult)
        self.assertEqual(result.model_id, "google/timesfm-2.5-200m-transformers")
        self.assertEqual(result.backend_name, "timesfm-2.5-transformers")
        self.assertEqual(result.symbol, "EURUSD")
        self.assertEqual(result.source_file, "EURUSD_2024_01.csv")
        self.assertEqual(result.context_window, 4)
        self.assertEqual(result.forecast_horizon, 2)
        self.assertEqual(result.actual_future.shape, (2,))


# ---------------------------------------------------------------------------
# 5. forecast_batch preserves order
# ---------------------------------------------------------------------------

class ForecastBatchTest(unittest.TestCase):

    def test_forecast_batch_returns_one_result_per_sample(self):
        stub_transformers, _ = _make_stub_transformers_module(np.zeros(2, dtype=np.float32))
        sample = _make_sample(context_window=4, forecast_horizon=2)

        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model()
            results = forecast_batch(handle, [sample, sample, sample])

        self.assertEqual(len(results), 3)

    def test_forecast_batch_preserves_sample_order(self):
        stub_transformers, _ = _make_stub_transformers_module(np.zeros(2, dtype=np.float32))
        s1 = _make_sample(context_window=4, forecast_horizon=2)
        s2 = _make_sample(context_window=4, forecast_horizon=3)

        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model()
            results = forecast_batch(handle, [s1, s2])

        self.assertEqual(results[0].forecast_horizon, 2)
        self.assertEqual(results[1].forecast_horizon, 3)


# ---------------------------------------------------------------------------
# 6. Custom model_id and device passthrough
# ---------------------------------------------------------------------------

class LoadModelOptionsTest(unittest.TestCase):

    def test_custom_model_id_is_preserved_in_handle(self):
        stub_transformers, _ = _make_stub_transformers_module(np.zeros(5))
        custom_id = "my-org/my-timesfm-2.5-ft"

        with patch.dict(sys.modules, {"transformers": stub_transformers}):
            handle = load_model(
                model_id=custom_id,
                backend_name="timesfm-2.5-transformers",
            )

        self.assertEqual(handle.model_id, custom_id)

    def test_missing_transformers_raises_import_error(self):
        with patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises((ImportError, AttributeError, TypeError)):
                load_model(backend_name="timesfm-2.5-transformers")


if __name__ == "__main__":
    unittest.main()
