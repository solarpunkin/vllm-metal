# SPDX-License-Identifier: Apache-2.0
"""Batched KV Cache utilities for MLX-based inference.

Provides utilities for managing multiple KV caches as a batch for efficient
parallel processing of decode requests.
"""

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class BatchedCacheManager:
    """Manages batched KV cache operations for efficient decode.

    This class provides utilities for:
    - Merging multiple individual KV caches into a batch
    - Running batched forward passes
    - Extracting individual caches after batched operations
    """

    def __init__(self, model: Any):
        """Initialize the batch cache manager.

        Args:
            model: The MLX model to create caches for
        """
        self.model = model
        self._batch_cache: list[Any] | None = None
        self._batch_indices: dict[str, int] = {}

    @staticmethod
    def merge_caches(caches: Sequence[list[Any]]) -> list[Any] | None:
        """Merge multiple prompt caches into a batched cache.

        This uses MLX's BatchKVCache.merge() pattern to combine individual
        caches into a single batch-aware cache.

        Args:
            caches: List of prompt caches (each cache is a list of per-layer caches)

        Returns:
            Batched cache, or None if no caches provided
        """
        if not caches:
            return None

        # Import BatchKVCache from mlx_lm
        try:
            from mlx_lm.models.cache import BatchKVCache
        except ImportError:
            # Fallback: can't batch, return None
            return None

        num_layers = len(caches[0])
        merged_cache = []

        for layer_idx in range(num_layers):
            # Collect per-layer caches
            layer_caches = [cache[layer_idx] for cache in caches]

            # Merge into batch cache for this layer
            if all(isinstance(c, KVCache) for c in layer_caches):
                batch_layer_cache = BatchKVCache.merge(layer_caches)
                merged_cache.append(batch_layer_cache)
            else:
                # Can't merge non-KVCache types
                return None

        return merged_cache

    @staticmethod
    def extract_cache(batch_cache: list[Any], index: int) -> list[Any]:
        """Extract a single cache from a batched cache.

        Args:
            batch_cache: The batched cache
            index: Index of the cache to extract

        Returns:
            Individual cache for the given index
        """
        try:
            from mlx_lm.models.cache import BatchKVCache
        except ImportError:
            raise RuntimeError("BatchKVCache not available") from None

        extracted = []
        for layer_cache in batch_cache:
            if isinstance(layer_cache, BatchKVCache):
                extracted.append(layer_cache.extract(index))
            else:
                # Can't extract from non-batch cache
                raise ValueError(f"Cannot extract from {type(layer_cache)}")

        return extracted

    def prepare_batch_decode(
        self, req_ids: list[str], caches: list[list[Any]]
    ) -> list[Any] | None:
        """Prepare a batch cache for decode operations.

        Args:
            req_ids: List of request IDs in batch order
            caches: List of individual caches

        Returns:
            Merged batch cache, or None if batching not possible
        """
        self._batch_cache = self.merge_caches(caches)
        if self._batch_cache is not None:
            self._batch_indices = {req_id: i for i, req_id in enumerate(req_ids)}
        return self._batch_cache

    def update_individual_caches(
        self, req_ids: list[str], original_caches: list[list[Any]]
    ) -> None:
        """Update individual caches from the batch cache state.

        After a batched forward pass, this extracts the updated state
        back to individual caches.

        Args:
            req_ids: List of request IDs
            original_caches: Original individual caches to update
        """
        if self._batch_cache is None:
            return

        for req_id, cache in zip(req_ids, original_caches, strict=True):
            idx = self._batch_indices.get(req_id)
            if idx is not None:
                # Extract updated cache state
                extracted = self.extract_cache(self._batch_cache, idx)
                # Update the original cache in place
                for i, layer_cache in enumerate(cache):
                    if hasattr(layer_cache, "state") and hasattr(extracted[i], "state"):
                        layer_cache.state = extracted[i].state


def create_left_padded_batch(
    token_sequences: list[list[int]], pad_token: int = 0
) -> tuple[mx.array, list[int]]:
    """Create a left-padded batch from variable-length sequences.

    Args:
        token_sequences: List of token ID sequences
        pad_token: Token to use for padding (default: 0)

    Returns:
        Tuple of (padded_batch, left_padding_amounts)
        - padded_batch: Shape (batch_size, max_length)
        - left_padding_amounts: How much each sequence was padded
    """
    if not token_sequences:
        return mx.array([], dtype=mx.int32), []

    max_length = max(len(seq) for seq in token_sequences)

    # Create padded array
    padded = []
    padding_amounts = []

    for seq in token_sequences:
        pad_amount = max_length - len(seq)
        padding_amounts.append(pad_amount)
        padded_seq = [pad_token] * pad_amount + list(seq)
        padded.append(padded_seq)

    return mx.array(padded, dtype=mx.int32), padding_amounts


def group_by_length(
    sequences: list[tuple[str, list[int]]],
    max_padding_ratio: float = 0.2,
    max_batch_size: int = 32,
) -> list[list[tuple[str, list[int]]]]:
    """Group sequences by similar length for efficient batching.

    Args:
        sequences: List of (req_id, token_ids) tuples
        max_padding_ratio: Maximum ratio of padding to sequence length
        max_batch_size: Maximum sequences per batch

    Returns:
        List of batches, where each batch contains sequences of similar length
    """
    if not sequences:
        return []

    # Sort by length
    sorted_seqs = sorted(sequences, key=lambda x: len(x[1]))

    batches: list[list[tuple[str, list[int]]]] = []
    current_batch: list[tuple[str, list[int]]] = []
    current_max_len = 0

    for req_id, tokens in sorted_seqs:
        seq_len = len(tokens)

        if current_batch:
            # Check if adding this sequence would exceed padding ratio
            new_max_len = max(current_max_len, seq_len)
            min_len_in_batch = len(current_batch[0][1])
            padding_ratio = (new_max_len - min_len_in_batch) / max(min_len_in_batch, 1)

            if (
                padding_ratio > max_padding_ratio
                or len(current_batch) >= max_batch_size
            ):
                # Start a new batch
                batches.append(current_batch)
                current_batch = []
                current_max_len = 0

        current_batch.append((req_id, tokens))
        current_max_len = max(current_max_len, seq_len)

    if current_batch:
        batches.append(current_batch)

    return batches
