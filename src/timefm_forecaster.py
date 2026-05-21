"""TimesFM 2.5 Transformers-based forecasting wrapper.

Replaces the retired v1 JAX/PAX path.

Public API (stable):
    load_model(model_id, backend_name, device, torch_dtype)
    forecast_sample(model, sample) -> TimeFMForecastResult
    forecast_batch(model, samples) -> list[TimeFMForecastResult]

Backends:
    "timesfm-2.5-transformers"  -- real model via HuggingFace Transformers
    "fake-timesfm"              -- unit-test stub, no torch required

The v1 "timesfm" backend (TimesFm JAX/PAX) is retired; passing
backend_name="timesfm" now raises ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.timefm_dataset import TimeFMSample

TIMESFM_25_MODEL_ID = "google/timesfm-2.5-200m-transformers"
TIMESFM_25_BACKEND = "timesfm-2.5-transformers"


# ---------------------------------------------------------------------------
# Fake backend (test-only, no torch dependency)
# ---------------------------------------------------------------------------

class FakeTimesFMBackend:
    """Last-value naive forecast backend for unit tests."""
    backend_name = "fake-timesfm"

    def predict(self, context: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        last_value = float(context[-1]) if len(context) else 0.0
        mean = np.full(horizon, last_value, dtype=float)
        return {
            "mean": mean,
            "lower": mean - 0.1,
            "upper": mean + 0.1,
        }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeFMModelHandle:
    model_id: str
    backend_name: str
    model: object


@dataclass(frozen=True)
class TimeFMForecastResult:
    model_id: str
    backend_name: str
    symbol: str
    source_file: str
    context_window: int
    forecast_horizon: int
    forecast_start_time: str
    forecast_end_time: str
    actual_future: np.ndarray
    predicted_mean: np.ndarray
    lower_quantile: np.ndarray | None
    upper_quantile: np.ndarray | None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = TIMESFM_25_MODEL_ID,
    backend_name: str = TIMESFM_25_BACKEND,
    device: str = "cpu",
    torch_dtype: str | None = None,
) -> TimeFMModelHandle:
    """Load a TimesFM model and return a handle.

    Args:
        model_id:     HuggingFace model id or local path.
        backend_name: "timesfm-2.5-transformers" for real inference,
                      "fake-timesfm" for unit tests.
        device:       "cpu", "mps", "cuda", or "auto".
        torch_dtype:  Optional dtype string e.g. "bfloat16". None uses model default.

    Returns:
        TimeFMModelHandle wrapping the loaded model object.

    Raises:
        ValueError:   if an unsupported or retired backend_name is given.
        ImportError:  if the transformers package is not installed.
    """
    backend_name = str(backend_name)

    if backend_name == "fake-timesfm":
        return TimeFMModelHandle(
            model_id=model_id,
            backend_name=backend_name,
            model=FakeTimesFMBackend(),
        )

    if backend_name == "timesfm":
        raise ValueError(
            "The v1 'timesfm' backend (JAX/PAX/Praxis) is retired for this project. "
            "Use backend_name='timesfm-2.5-transformers' with the HuggingFace Transformers stack."
        )

    if backend_name == "timesfm-2.5-transformers":
        try:
            from transformers import TimesFm2_5ModelForPrediction  # type: ignore
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "transformers package is required for TimesFM 2.5 inference. "
                "Install it with: pip install transformers accelerate"
            ) from exc

        kwargs: dict = {}
        if torch_dtype is not None:
            import torch
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch_dtype)

        # device_map="auto" triggers accelerate; for cpu/mps pass device_map directly
        if device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = device

        model = TimesFm2_5ModelForPrediction.from_pretrained(model_id, **kwargs)
        model.eval()
        return TimeFMModelHandle(
            model_id=model_id,
            backend_name=backend_name,
            model=model,
        )

    raise ValueError(
        f"Unsupported backend_name: {backend_name!r}. "
        f"Choose 'timesfm-2.5-transformers' or 'fake-timesfm'."
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _coerce_to_numpy_1d(tensor_like, length: int) -> np.ndarray:
    """Convert a tensor-like [1, N] or [N] to a 1-D numpy float array of `length`."""
    if hasattr(tensor_like, "numpy"):
        arr = tensor_like.numpy()
    elif hasattr(tensor_like, "detach"):
        arr = tensor_like.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_like, dtype=float)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]          # drop batch dim
    return arr[:length]


def _predict_with_fake_backend(
    backend: FakeTimesFMBackend,
    context: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    prediction = backend.predict(context=context, horizon=horizon)
    mean = np.asarray(prediction["mean"], dtype=float)
    lower_raw = prediction.get("lower")
    upper_raw = prediction.get("upper")
    lower = np.asarray(lower_raw, dtype=float) if lower_raw is not None else None
    upper = np.asarray(upper_raw, dtype=float) if upper_raw is not None else None
    return mean, lower, upper


def _predict_with_transformers_model(
    raw_model: object,
    context: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run inference with a TimesFm2_5ModelForPrediction instance.

    Calls model(past_values=...) following the official API.
    Returns (mean, lower, upper) as 1-D numpy arrays of length `horizon`.
    """
    # Try to use torch if available, otherwise keep as numpy for stubbed tests.
    try:
        import torch
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # [1, T]
    except ImportError:
        # Running under a stub in tests — pass the raw array and let the stub handle it
        context_tensor = context[np.newaxis, :]  # [1, T]

    context_len = context.shape[0]
    output = raw_model(
        past_values=context_tensor,
        forecast_context_len=context_len,
    )

    mean_tensor = output.mean_predictions  # [batch, max_output_len]
    # Grab batch 0 and trim to the requested horizon
    if hasattr(mean_tensor, "__getitem__"):
        row = mean_tensor[0]
    else:
        row = mean_tensor
    mean = _coerce_to_numpy_1d(row, horizon)

    return mean, None, None  # quantile outputs not exposed in base Transformers path


def _predict_with_backend(
    model: TimeFMModelHandle,
    context: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    backend = model.model

    if model.backend_name == "fake-timesfm":
        return _predict_with_fake_backend(backend, context, horizon)

    if model.backend_name == "timesfm-2.5-transformers":
        return _predict_with_transformers_model(backend, context, horizon)

    raise TypeError(
        f"Unsupported backend '{model.backend_name}' on model object {type(backend)!r}"
    )


# ---------------------------------------------------------------------------
# Public forecast functions
# ---------------------------------------------------------------------------

def forecast_sample(model: TimeFMModelHandle, sample: TimeFMSample) -> TimeFMForecastResult:
    """Run a single-sample forecast and return a structured result."""
    context = np.asarray(sample.target_context, dtype=float)
    predicted_mean, lower_quantile, upper_quantile = _predict_with_backend(
        model,
        context=context,
        horizon=sample.forecast_horizon,
    )
    return TimeFMForecastResult(
        model_id=model.model_id,
        backend_name=model.backend_name,
        symbol=sample.symbol,
        source_file=sample.source_file,
        context_window=sample.context_window,
        forecast_horizon=sample.forecast_horizon,
        forecast_start_time=str(sample.forecast_start_time),
        forecast_end_time=str(sample.forecast_end_time),
        actual_future=np.asarray(sample.target_future, dtype=float),
        predicted_mean=predicted_mean,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def forecast_batch(
    model: TimeFMModelHandle,
    samples: Iterable[TimeFMSample],
) -> list[TimeFMForecastResult]:
    """Run forecast_sample over a list of samples, preserving order."""
    return [forecast_sample(model, sample) for sample in samples]
