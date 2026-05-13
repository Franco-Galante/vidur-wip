import random
from typing import Tuple

import numpy as np
import pandas as pd

from vidur.config import CustomRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)


class CustomRequestLengthGenerator(BaseRequestLengthGenerator):
    """Length generator that samples (with replacement) from a CSV file.

    The CSV must contain two columns: ``num_prefill_tokens`` and
    ``num_decode_tokens``.  Each call to ``get_next_num_tokens`` picks a row
    uniformly at random, making the CSV define an empirical joint distribution.
    This differs from :class:`TraceRequestLengthGenerator`, which replays rows
    sequentially and stops when the trace is exhausted.

    Pair lengths whose sum exceeds ``max_tokens`` are clipped proportionally,
    following the same logic as the trace generator.
    """

    def __init__(self, config: CustomRequestLengthGeneratorConfig):
        super().__init__(config)

        df = pd.read_csv(config.trace_file)

        # Apply max_tokens cap proportionally (same logic as TraceRequestLengthGenerator).
        total = df["num_prefill_tokens"] + df["num_decode_tokens"]
        excess = (total - config.max_tokens).clip(lower=0)
        prefill_ratio = df["num_prefill_tokens"] / total
        decode_ratio = df["num_decode_tokens"] / total

        df["num_prefill_tokens"] = (
            df["num_prefill_tokens"] - np.ceil(excess * prefill_ratio)
        ).astype(int)
        df["num_decode_tokens"] = (
            df["num_decode_tokens"] - np.ceil(excess * decode_ratio)
        ).astype(int)

        df["num_prefill_tokens"] = df["num_prefill_tokens"].clip(lower=1)
        df["num_decode_tokens"] = df["num_decode_tokens"].clip(lower=1)

        assert all(df["num_prefill_tokens"] > 0), "All prefill token counts must be positive after clipping."
        assert all(df["num_decode_tokens"] > 0), "All decode token counts must be positive after clipping."

        assert all(
            df["num_prefill_tokens"] + df["num_decode_tokens"] <= config.max_tokens
        ), "Some rows still exceed max_tokens after clipping."

        self._prefill_tokens = df["num_prefill_tokens"].tolist()
        self._decode_tokens = df["num_decode_tokens"].tolist()
        self._rng = np.random.default_rng(config.seed)

    def get_next_num_tokens(self) -> Tuple[float, float]:
        idx = int(self._rng.integers(len(self._prefill_tokens)))
        return float(self._prefill_tokens[idx]), float(self._decode_tokens[idx])
