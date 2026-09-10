"""
End-to-end test of generate_embeddings.run() against the fixture built from the real uploaded
sh1.json. Uses:
  - a REAL tokenizers.Tokenizer (BPE, trained on the fixture text itself) — exercises the actual
    tokenizers library API this script depends on, not a mock of it.
  - a MOCK onnxruntime.InferenceSession — standing in for the real ~2GB Qwen3-Embedding model,
    which isn't available in this environment (no network access to Hugging Face here). Returns
    deterministic synthetic hidden states shaped correctly for the real batch/seq_len/dimension,
    so this test verifies the *orchestration* (batching, tokenization, pooling call, checkpoint/
    resume, output file format) is correct — it cannot and does not claim the real model's
    output would be semantically good, only that this script would use it correctly.
"""
import json
import os
import shutil
import struct
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")
import numpy as np  # noqa: E402
from tokenizers import Tokenizer, models, trainers, pre_tokenizers  # noqa: E402
import generate_embeddings  # noqa: E402

FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ganjoor-data-fixture")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "_test_output")
HIDDEN_DIM = 8  # small, arbitrary - a real Qwen3-Embedding-0.6B is 1024, doesn't matter for this test


def _build_tiny_tokenizer():
    """A real BPE tokenizer, trained on the fixture's own text, just so encode_batch() runs
    against genuine tokenizers-library behavior instead of a hand-rolled fake."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>"], vocab_size=300)

    with open(os.path.join(FIXTURE_ROOT, "poets", "hafez", "ghazal", "sh1.json"), encoding="utf-8") as f:
        summary_text = json.load(f)["PoemSummary"]

    tokenizer.train_from_iterator([summary_text, "یک متن دیگر برای آموزش", "another sentence to train on"], trainer)
    return tokenizer


class MockOnnxSession:
    """Stands in for onnxruntime.InferenceSession. Declares the SAME kind of KV-cache input
    signature the real model turned out to have (position_ids + past_key_values.N.key/.value per
    layer, confirmed via a real --inspect-only run against the actual downloaded model) — using a
    plain simple signature here would let a bug in build_extra_inputs() pass silently. run()
    asserts every declared input is actually present in the feed dict, the same way real ONNX
    Runtime would refuse to run with a missing input, so a regression here fails the test instead
    of only failing months from now against the real model."""

    NUM_LAYERS = 2
    NUM_KV_HEADS = 3
    HEAD_DIM = 4

    def get_inputs(self):
        class FakeInput:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape
                self.type = "tensor(int64)" if "past_key_values" not in name and name != "position_ids" else (
                    "tensor(float)" if "past_key_values" in name else "tensor(int64)"
                )

        inputs = [
            FakeInput("input_ids", ["batch_size", "sequence_length"]),
            FakeInput("attention_mask", ["batch_size", "total_sequence_length"]),
            FakeInput("position_ids", ["batch_size", "sequence_length"]),
        ]
        for layer in range(self.NUM_LAYERS):
            inputs.append(FakeInput(f"past_key_values.{layer}.key", ["batch_size", self.NUM_KV_HEADS, "past_sequence_length", self.HEAD_DIM]))
            inputs.append(FakeInput(f"past_key_values.{layer}.value", ["batch_size", self.NUM_KV_HEADS, "past_sequence_length", self.HEAD_DIM]))
        return inputs

    def get_outputs(self):
        class FakeOutput:
            def __init__(self, name):
                self.name = name
                self.shape = ["batch", "seq", HIDDEN_DIM]
                self.type = "tensor(float)"
        return [FakeOutput("last_hidden_state")]

    def run(self, output_names, feed_dict):
        expected_names = {inp.name for inp in self.get_inputs()}
        missing = expected_names - set(feed_dict.keys())
        if missing:
            raise KeyError(f"missing required model input(s), just like real ONNX Runtime would refuse to run without them: {missing}")

        input_ids = feed_dict["input_ids"]
        batch, seq_len = input_ids.shape
        # deterministic-but-input-dependent fake hidden states: token id broadcast across a
        # small hidden dim, so pooling on different positions actually yields different vectors
        hidden = np.zeros((batch, seq_len, HIDDEN_DIM), dtype=np.float32)
        for b in range(batch):
            for s in range(seq_len):
                hidden[b, s, :] = (input_ids[b, s] % 97) + np.arange(HIDDEN_DIM)
        return [hidden]


def _clean_output_dir():
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


def test_full_run_produces_correct_output_files():
    _clean_output_dir()
    tokenizer = _build_tiny_tokenizer()

    args = generate_embeddings.parse_args.__wrapped__ if False else None  # not used; build args directly below
    args = type("Args", (), {})()
    args.source = FIXTURE_ROOT
    args.model_dir = "/unused-in-this-test"
    args.output = TEST_OUTPUT_DIR
    args.model_name = "test-model"
    args.model_file = "model.onnx"
    args.batch_size = 16
    args.max_length = 64
    args.limit = None
    args.resume = False
    args.gpu = False
    args.inspect_only = False
    args.input_ids_name = "input_ids"
    args.attention_mask_name = "attention_mask"
    args.output_name = "last_hidden_state"

    with patch("onnxruntime.InferenceSession", return_value=MockOnnxSession()), \
         patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), \
         patch("os.path.isfile", side_effect=lambda p: True):  # skip the model.onnx/tokenizer.json existence check
        generate_embeddings.run(args)

    # only sh1 has a non-empty PoemSummary (sh2's is ""), so exactly 1 poem should be embedded
    index_path = os.path.join(TEST_OUTPUT_DIR, "embeddings-index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    assert index["count"] == 1, f"expected 1 embedded poem, got {index['count']}"
    assert index["ids"] == [2130], f"expected poem id 2130 (sh1), got {index['ids']}"
    assert index["dimension"] == HIDDEN_DIM
    assert index["normalized"] is True
    assert index["poolingMethod"] == "last_token"

    bin_path = os.path.join(TEST_OUTPUT_DIR, "embeddings.f32")
    file_size = os.path.getsize(bin_path)
    expected_size = index["count"] * index["dimension"] * 4  # float32 = 4 bytes
    assert file_size == expected_size, f"embeddings.f32 is {file_size} bytes, expected {expected_size}"

    # read the one vector back and confirm it's unit-normalized (real property, not mocked)
    with open(bin_path, "rb") as f:
        raw = f.read()
    vector = np.array(struct.unpack(f"<{HIDDEN_DIM}f", raw), dtype=np.float32)
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-5, f"expected unit-norm vector, got norm={norm}"

    print("test_full_run_produces_correct_output_files: PASS")
    return args


def test_resume_skips_already_completed_poems():
    # re-run with --resume against the output of the previous test; since sh1 (the only
    # embeddable poem) is already in the checkpoint, this run should embed nothing new
    _, tokenizer = None, _build_tiny_tokenizer()

    args = type("Args", (), {})()
    args.source = FIXTURE_ROOT
    args.model_dir = "/unused-in-this-test"
    args.output = TEST_OUTPUT_DIR
    args.model_name = "test-model"
    args.model_file = "model.onnx"
    args.batch_size = 16
    args.max_length = 64
    args.limit = None
    args.resume = True
    args.gpu = False
    args.inspect_only = False
    args.input_ids_name = "input_ids"
    args.attention_mask_name = "attention_mask"
    args.output_name = "last_hidden_state"

    checkpoint_path = os.path.join(TEST_OUTPUT_DIR, "checkpoint.ndjson")
    assert os.path.exists(checkpoint_path), "expected a checkpoint from the previous test run"

    with patch("onnxruntime.InferenceSession", return_value=MockOnnxSession()), \
         patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), \
         patch("os.path.isfile", side_effect=lambda p: True):
        generate_embeddings.run(args)

    with open(os.path.join(TEST_OUTPUT_DIR, "embeddings-index.json"), encoding="utf-8") as f:
        index = json.load(f)
    # still exactly 1 - resume should not have duplicated or re-processed the already-done poem
    assert index["count"] == 1, f"expected resume to add nothing new, got count={index['count']}"
    # this is the specific bug this test was written to catch: a resume run with nothing new to
    # process must NOT overwrite the previous run's correctly-recorded dimension with None
    assert index["dimension"] == HIDDEN_DIM, f"resume corrupted dimension: expected {HIDDEN_DIM}, got {index['dimension']}"
    print("test_resume_skips_already_completed_poems: PASS")

    _clean_output_dir()


MULTI_FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ganjoor-data-fixture-multi")
MULTI_TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "_test_output_multi")


def _make_multi_args(output_dir, limit, resume):
    args = type("Args", (), {})()
    args.source = MULTI_FIXTURE_ROOT
    args.model_dir = "/unused-in-this-test"
    args.output = output_dir
    args.model_name = "test-model"
    args.model_file = "model.onnx"
    args.batch_size = 1  # force multiple separate batches, so an "interruption" between them is meaningful
    args.max_length = 64
    args.limit = limit
    args.resume = resume
    args.gpu = False
    args.inspect_only = False
    args.input_ids_name = "input_ids"
    args.attention_mask_name = "attention_mask"
    args.output_name = "last_hidden_state"
    return args


def test_interrupt_then_resume_produces_no_duplicate_or_misaligned_rows():
    """
    Specifically simulates what happens when a real run gets interrupted (e.g. Ctrl+C to shut
    the machine down) partway through and is later resumed — the exact scenario this script
    needs to get right, and the exact scenario a previous version of it got wrong (embeddings.f32
    and the checkpoint could fall out of sync, causing --resume to re-embed and duplicate rows
    that were already on disk). All 4 poems in this fixture have a summary, so a correct run
    embeds all 4 exactly once no matter how it's split across an interruption.
    """
    if os.path.exists(MULTI_TEST_OUTPUT_DIR):
        shutil.rmtree(MULTI_TEST_OUTPUT_DIR)
    tokenizer = _build_tiny_tokenizer()

    # "run 1": process only the first 2 of 4 poems, simulating being interrupted right after them
    args_first = _make_multi_args(MULTI_TEST_OUTPUT_DIR, limit=2, resume=False)
    with patch("onnxruntime.InferenceSession", return_value=MockOnnxSession()), \
         patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), \
         patch("os.path.isfile", side_effect=lambda p: True):
        generate_embeddings.run(args_first)

    with open(os.path.join(MULTI_TEST_OUTPUT_DIR, "embeddings-index.json"), encoding="utf-8") as f:
        index_after_first = json.load(f)
    assert index_after_first["count"] == 2, f"expected 2 after first (interrupted) run, got {index_after_first['count']}"

    # "run 2": resume, now against the full (unlimited) poem list - should embed only the 2 that
    # weren't done yet, and never touch the 2 already-embedded ones again
    args_resume = _make_multi_args(MULTI_TEST_OUTPUT_DIR, limit=None, resume=True)
    with patch("onnxruntime.InferenceSession", return_value=MockOnnxSession()), \
         patch("tokenizers.Tokenizer.from_file", return_value=tokenizer), \
         patch("os.path.isfile", side_effect=lambda p: True):
        generate_embeddings.run(args_resume)

    with open(os.path.join(MULTI_TEST_OUTPUT_DIR, "embeddings-index.json"), encoding="utf-8") as f:
        final_index = json.load(f)

    assert final_index["count"] == 4, f"expected all 4 poems embedded after resume, got {final_index['count']}"
    assert sorted(final_index["ids"]) == [2130, 2131, 2132, 2133], f"unexpected ids: {final_index['ids']}"
    assert len(final_index["ids"]) == len(set(final_index["ids"])), f"duplicate ids in final index: {final_index['ids']}"
    # a stronger check than the sorted() one above: the first 2 entries must be exactly what run 1
    # produced, in that exact order, followed by exactly what the resume run added, in order.
    # This is what actually catches the ordering bug this test is named for — rebuilding all_ids
    # from a *set* on resume (an earlier version of this script did) can silently reshuffle which
    # id is claimed to go with which row, and a plain sorted()-based check wouldn't notice if the
    # reshuffling happened to still end up numerically sorted.
    assert final_index["ids"][:2] == [2130, 2131], f"run 1's ids/order not preserved: {final_index['ids']}"
    assert final_index["ids"][2:] == [2132, 2133], f"resume's ids/order wrong: {final_index['ids']}"

    # the real check this test exists for: exactly 4 rows in the binary file, not 6 (which is
    # what the old buggy version would have produced if the checkpoint had lagged behind what
    # was actually written to embeddings.f32 at the moment of "interruption")
    bin_path = os.path.join(MULTI_TEST_OUTPUT_DIR, "embeddings.f32")
    file_size = os.path.getsize(bin_path)
    expected_size = 4 * HIDDEN_DIM * 4  # 4 poems x HIDDEN_DIM float32s x 4 bytes/float
    assert file_size == expected_size, (
        f"embeddings.f32 is {file_size} bytes (expected {expected_size} for exactly 4 rows) — "
        f"a mismatch here means duplicate/orphaned rows got written across the interruption"
    )

    checkpoint_path = os.path.join(MULTI_TEST_OUTPUT_DIR, "checkpoint.ndjson")
    with open(checkpoint_path, encoding="utf-8") as f:
        checkpoint_ids = [int(line) for line in f if line.strip()]
    assert sorted(checkpoint_ids) == [2130, 2131, 2132, 2133], f"unexpected checkpoint contents: {checkpoint_ids}"
    assert len(checkpoint_ids) == len(set(checkpoint_ids)), f"duplicate ids in checkpoint: {checkpoint_ids}"

    print("test_interrupt_then_resume_produces_no_duplicate_or_misaligned_rows: PASS")
    shutil.rmtree(MULTI_TEST_OUTPUT_DIR)


if __name__ == "__main__":
    test_full_run_produces_correct_output_files()
    test_resume_skips_already_completed_poems()
    test_interrupt_then_resume_produces_no_duplicate_or_misaligned_rows()
    print("\nAll generate_embeddings integration tests passed.")
