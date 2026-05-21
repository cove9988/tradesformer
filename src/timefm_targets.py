from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize_future_targets(
    future_values: Iterable[float],
    positive_threshold: float = 0.0,
    negative_threshold: float = 0.0,
) -> dict[str, float | int | str]:
    values = np.asarray(list(future_values), dtype=float)
    if values.size == 0:
        raise ValueError("future_values must not be empty")

    future_return_sum = float(values.sum())
    future_abs_move = float(np.abs(values).sum())

    if future_return_sum > positive_threshold:
        future_direction = "up"
        future_direction_class = 1
    elif future_return_sum < negative_threshold:
        future_direction = "down"
        future_direction_class = -1
    else:
        future_direction = "flat"
        future_direction_class = 0

    return {
        "future_return_sum": future_return_sum,
        "future_abs_move": future_abs_move,
        "future_direction": future_direction,
        "future_direction_class": future_direction_class,
    }
