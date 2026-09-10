"""
Pooling and normalization for Qwen3-Embedding models.

Split out from generate_embeddings.py specifically so it can be unit-tested with synthetic
arrays, independent of having the actual ~2GB ONNX model downloaded. This is the one piece of
the whole pipeline that's easy to get subtly wrong and hard to notice — an incorrect pooling
strategy doesn't crash, it just silently produces embeddings that don't match how the model was
trained, which shows up later as "search results are mysteriously mediocre," not as an error.

Per the model's documentation (Qwen3-Embedding family, and confirmed on the ONNX export cards
for this specific model): pooling and normalization are NOT part of the ONNX graph itself — the
graph only returns per-token hidden states. The caller must:
  1. Take the hidden state of the LAST NON-PADDING token for each sequence (not mean-pooling,
     not the first/CLS token — this is a decoder-style model, not BERT).
  2. L2-normalize that vector.
After that, cosine similarity between two embeddings is just their dot product.
"""

import numpy as np


def last_token_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    hidden_states: (batch, seq_len, hidden_dim) float array — the ONNX model's raw output.
    attention_mask: (batch, seq_len) int array — 1 for real tokens, 0 for padding.

    Returns (batch, hidden_dim): the hidden state at each sequence's last real (non-padding)
    token position. Works regardless of whether the tokenizer left-pads or right-pads, by
    actually computing each row's last real-token index from its own attention mask rather than
    assuming a fixed padding side.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"expected hidden_states with shape (batch, seq_len, hidden_dim), got {hidden_states.shape}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} doesn't match hidden_states' "
            f"(batch, seq_len) shape {hidden_states.shape[:2]}"
        )

    batch_size = hidden_states.shape[0]
    # index of the last 1 in each row of the attention mask
    last_real_token_idx = attention_mask.shape[1] - 1 - np.argmax(attention_mask[:, ::-1], axis=1)

    # guard against a fully-empty row (attention_mask all zeros) — shouldn't happen with real
    # tokenized input, but fail loudly with a clear message instead of silently returning
    # position 0's hidden state, which would be wrong and hard to trace back to this cause.
    if np.any(attention_mask.sum(axis=1) == 0):
        bad_rows = np.where(attention_mask.sum(axis=1) == 0)[0].tolist()
        raise ValueError(f"attention_mask is all-zero for batch row(s) {bad_rows} — empty input text?")

    return hidden_states[np.arange(batch_size), last_real_token_idx, :]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    vectors: (batch, hidden_dim). Returns unit-length vectors along the last axis, so that
    cosine similarity between any two rows reduces to a plain dot product.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def embed_pooled_and_normalized(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Convenience wrapper: last-token pooling, then L2 normalization, in one call."""
    return l2_normalize(last_token_pool(hidden_states, attention_mask))
