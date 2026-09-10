"""
Tests for pooling.py using synthetic hidden states — no real ONNX model needed. These verify the
pooling math itself is correct; they cannot verify the real Qwen3-Embedding model produces good
embeddings (that needs the actual model, which isn't available in this environment).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")
from pooling import last_token_pool, l2_normalize, embed_pooled_and_normalized  # noqa: E402


def test_last_token_pool_right_padded():
    # 2 sequences, seq_len=4, hidden_dim=3. Row 0 has 4 real tokens (no padding), row 1 has
    # 2 real tokens then 2 padding positions (right-padded, the common case).
    hidden_states = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[9, 9, 9], [8, 8, 8], [0, 0, 0], [0, 0, 0]],
    ], dtype=np.float32)
    attention_mask = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=np.int64)

    result = last_token_pool(hidden_states, attention_mask)

    # row 0's last real token is position 3 -> [4,4,4]
    assert np.array_equal(result[0], [4, 4, 4]), f"row 0 wrong: {result[0]}"
    # row 1's last real token is position 1 -> [8,8,8], NOT position 3 (padding)
    assert np.array_equal(result[1], [8, 8, 8]), f"row 1 wrong: {result[1]}"
    print("test_last_token_pool_right_padded: PASS")


def test_last_token_pool_left_padded():
    # left-padded: row 1 has padding at the START, real tokens at the end
    hidden_states = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[0, 0, 0], [0, 0, 0], [7, 7, 7], [6, 6, 6]],
    ], dtype=np.float32)
    attention_mask = np.array([
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.int64)

    result = last_token_pool(hidden_states, attention_mask)

    assert np.array_equal(result[0], [4, 4, 4]), f"row 0 wrong: {result[0]}"
    # row 1's last real token is position 3 -> [6,6,6]
    assert np.array_equal(result[1], [6, 6, 6]), f"row 1 wrong: {result[1]}"
    print("test_last_token_pool_left_padded: PASS")


def test_last_token_pool_rejects_all_zero_row():
    hidden_states = np.zeros((1, 3, 2), dtype=np.float32)
    attention_mask = np.array([[0, 0, 0]], dtype=np.int64)
    try:
        last_token_pool(hidden_states, attention_mask)
        raise AssertionError("expected ValueError for all-zero attention mask row")
    except ValueError as e:
        assert "row(s)" in str(e)
        print("test_last_token_pool_rejects_all_zero_row: PASS")


def test_l2_normalize_unit_length():
    vectors = np.array([[3, 4, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    result = l2_normalize(vectors)

    norms = np.linalg.norm(result, axis=-1)
    assert np.isclose(norms[0], 1.0), f"expected unit norm, got {norms[0]}"
    assert np.isclose(norms[1], 1.0), f"expected unit norm, got {norms[1]}"
    # zero vector should not produce NaN (division-by-zero guard)
    assert not np.any(np.isnan(result[2])), f"zero-vector row produced NaN: {result[2]}"
    print("test_l2_normalize_unit_length: PASS")


def test_l2_normalize_cosine_via_dot_product():
    # after normalization, dot product of two rows should equal their cosine similarity
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, -1.0, 2.0])
    expected_cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    normalized = l2_normalize(np.stack([a, b]))
    dot_product = np.dot(normalized[0], normalized[1])

    assert np.isclose(dot_product, expected_cosine, atol=1e-6), (
        f"dot product {dot_product} != expected cosine {expected_cosine}"
    )
    print("test_l2_normalize_cosine_via_dot_product: PASS")


def test_embed_pooled_and_normalized_end_to_end():
    hidden_states = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[9, 9, 9], [8, 8, 8], [0, 0, 0], [0, 0, 0]],
    ], dtype=np.float32)
    attention_mask = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=np.int64)

    result = embed_pooled_and_normalized(hidden_states, attention_mask)

    assert result.shape == (2, 3)
    norms = np.linalg.norm(result, axis=-1)
    assert np.allclose(norms, 1.0), f"expected unit-norm rows, got {norms}"
    # row 0 pooled to [4,4,4] before normalization -> direction should be [1,1,1]/sqrt(3)
    expected_row0_direction = np.array([4, 4, 4]) / np.linalg.norm([4, 4, 4])
    assert np.allclose(result[0], expected_row0_direction, atol=1e-6)
    print("test_embed_pooled_and_normalized_end_to_end: PASS")


if __name__ == "__main__":
    test_last_token_pool_right_padded()
    test_last_token_pool_left_padded()
    test_last_token_pool_rejects_all_zero_row()
    test_l2_normalize_unit_length()
    test_l2_normalize_cosine_via_dot_product()
    test_embed_pooled_and_normalized_end_to_end()
    print("\nAll pooling tests passed.")
