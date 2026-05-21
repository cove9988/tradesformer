import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from src.timesfm_loader_spike import build_spike_report
from src.run_timesfm_forecasts import run_forecasts_for_split
from src.timesfm_split import build_split_manifest, write_split_manifest


class FakeBackend:
    backend_name = "fake-timesfm"

    def predict(self, context: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        last_value = float(context[-1]) if len(context) else 0.0
        mean = np.full(horizon, last_value)
        return {
            "mean": mean,
            "lower": mean - 0.1,
            "upper": mean + 0.1,
        }


class TimeFMForecasterTest(unittest.TestCase):
    def _sample(self):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC"),
                "symbol": ["EURUSD"] * 8,
                "open": [1.1] * 8,
                "high": [1.11] * 8,
                "low": [1.09] * 8,
                "close": [1.1] * 8,
                "log_return_1": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
                "close_rel_week_open": list(range(8)),
                "hour_sin": [0.0] * 8,
            }
        )
        return build_timesfm_samples(
            frame,
            context_window=3,
            forecast_horizon=2,
            target_col="log_return_1",
            feature_cols=["close_rel_week_open", "hour_sin"],
            source_file="EURUSD_2024_01.csv",
        )[0]

    def test_forecast_sample_returns_structured_result_with_metadata(self):
        sample = self._sample()
        model = TimeFMModelHandle(model_id="fake/model", backend_name="fake-timesfm", model=FakeBackend())

        result = forecast_sample(model, sample)

        self.assertIsInstance(result, TimeFMForecastResult)
        self.assertEqual(result.model_id, "fake/model")
        self.assertEqual(result.backend_name, "fake-timesfm")
        self.assertEqual(result.context_window, 3)
        self.assertEqual(result.forecast_horizon, 2)
        self.assertEqual(result.source_file, "EURUSD_2024_01.csv")
        self.assertEqual(result.predicted_mean.shape, (2,))
        self.assertAlmostEqual(float(result.predicted_mean[0]), 0.002)
        self.assertAlmostEqual(float(result.lower_quantile[0]), -0.098)
        self.assertAlmostEqual(float(result.upper_quantile[0]), 0.102)

    def test_forecast_batch_preserves_order(self):
        sample = self._sample()
        model = TimeFMModelHandle(model_id="fake/model", backend_name="fake-timesfm", model=FakeBackend())

        results = forecast_batch(model, [sample, sample])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].forecast_start_time, results[1].forecast_start_time)


class TimeFMLoaderSpikeTest(unittest.TestCase):
    def test_build_spike_report_runs_with_explicit_fake_backend(self):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC"),
                "symbol": ["EURUSD"] * 8,
                "open": [1.1] * 8,
                "high": [1.11] * 8,
                "low": [1.09] * 8,
                "close": [1.1] * 8,
                "log_return_1": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
                "close_rel_week_open": list(range(8)),
                "hour_sin": [0.0] * 8,
            }
        )
        csv_path = Path(tempfile.gettempdir()) / "timesfm_loader_spike_sample.csv"
        frame.to_csv(csv_path, index=False)

        report = build_spike_report(
            csv_path=csv_path,
            target_col="log_return_1",
            feature_cols=["close_rel_week_open", "hour_sin"],
            context_window=3,
            forecast_horizon=2,
            backend_name="fake-timesfm",
        )

        self.assertEqual(report["backend_name"], "fake-timesfm")
        self.assertEqual(report["forecast_horizon"], 2)
        self.assertEqual(tuple(report["predicted_mean_shape"]), (2,))


class TimeFMRealBackendWiringTest(unittest.TestCase):
    @unittest.skip("v1 JAX/PAX backend retired — replaced by timesfm-2.5-transformers")
    def test_load_model_builds_pypi_timesfm_v1_backend_from_actual_api(self):
        class StubTimesFm:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs
                self.loaded_repo_id = None

            def load_from_checkpoint(self, *, repo_id, **kwargs):
                self.loaded_repo_id = repo_id
                self.load_kwargs = kwargs

        class StubModule:
            TimesFm = StubTimesFm

        with patch.dict(sys.modules, {"timesfm": StubModule()}):
            handle = load_model(model_id="google/timesfm-1.0-200m", backend_name="timesfm")

        self.assertEqual(handle.backend_name, "timesfm")
        self.assertEqual(handle.model_id, "google/timesfm-1.0-200m")
        self.assertIsInstance(handle.model, StubTimesFm)
        self.assertEqual(handle.model.init_kwargs["backend"], "cpu")
        self.assertEqual(handle.model.init_kwargs["per_core_batch_size"], 32)
        self.assertEqual(handle.model.init_kwargs["context_len"], 512)
        self.assertEqual(handle.model.init_kwargs["horizon_len"], 128)
        self.assertEqual(handle.model.init_kwargs["input_patch_len"], 32)
        self.assertEqual(handle.model.init_kwargs["output_patch_len"], 128)
        self.assertEqual(handle.model.init_kwargs["num_layers"], 20)
        self.assertEqual(handle.model.init_kwargs["model_dims"], 1280)
        self.assertEqual(handle.model.loaded_repo_id, "google/timesfm-1.0-200m")


class TimeFMActualForecastApiTest(unittest.TestCase):
    @unittest.skip("v1 JAX/PAX backend retired — replaced by timesfm-2.5-transformers")
    def test_forecast_sample_supports_timesfm_forecast_api(self):
        sample = TimeFMForecasterTest()._sample()

        class StubTimesFm:
            def forecast(self, inputs, freq=None):
                self.inputs = inputs
                self.freq = freq
                point = np.array([[0.11, 0.12]], dtype=float)
                full = np.array([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]], dtype=float)
                return point, full

        model = TimeFMModelHandle(model_id="google/timesfm-1.0-200m", backend_name="timesfm", model=StubTimesFm())

        result = forecast_sample(model, sample)

        self.assertTrue(np.allclose(model.model.inputs[0], sample.target_context))
        self.assertEqual(model.model.freq, [0])
        self.assertTrue(np.allclose(result.predicted_mean, np.array([0.11, 0.12])))
        self.assertTrue(np.allclose(result.lower_quantile, np.array([0.02, 0.05])))
        self.assertTrue(np.allclose(result.upper_quantile, np.array([0.03, 0.06])))


class RunTimeFMForecastsTest(unittest.TestCase):
    def _write_week(self, directory: Path, week: int):
        frame = pd.DataFrame(
            {
                "time": pd.date_range(f"2024-01-{week:02d}", periods=8, freq="5min", tz="UTC"),
                "symbol": ["EURUSD"] * 8,
                "open": [1.1] * 8,
                "high": [1.11] * 8,
                "low": [1.09] * 8,
                "close": [1.1] * 8,
                "log_return_1": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
                "close_rel_week_open": list(range(8)),
                "hour_sin": [0.0] * 8,
            }
        )
        frame.to_csv(directory / f"EURUSD_2024_{week:02d}.csv", index=False)

    def test_run_forecasts_for_split_writes_jsonl_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "train"
            data_dir.mkdir()
            for week in range(1, 5):
                self._write_week(data_dir, week)

            manifest = build_split_manifest(data_dir, symbol="EURUSD", validation_weeks=1, test_weeks=1)
            manifest_path = base / "manifest.json"
            write_split_manifest(manifest_path, manifest)

            written = run_forecasts_for_split(
                data_directory=data_dir,
                manifest_path=manifest_path,
                split_name="train",
                output_directory=base / "predictions",
                model=TimeFMModelHandle(model_id="fake/model", backend_name="fake-timesfm", model=FakeBackend()),
                target_col="log_return_1",
                feature_cols=["close_rel_week_open", "hour_sin"],
                context_windows=[3],
                forecast_horizons=[2],
            )

            self.assertEqual(len(written), 1)
            output_path = written[0]
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.name.endswith("context_3_horizon_2.jsonl"))
            first_line = output_path.read_text().splitlines()[0]
            self.assertIn('"model_id": "fake/model"', first_line)
            self.assertIn('"predicted_mean": [0.002, 0.002]', first_line)


if __name__ == "__main__":
    unittest.main()
