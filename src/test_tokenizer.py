# src/test_tokenizer.py

import tensorflow as tf
import pytest
from tokenizer import tokenize_sequence, decode_sequence, VOCAB as vocab

def test_tokenizer_roundtrip():
    """Pytest-compatible roundtrip test for a valid sequence"""
    seq = "MFSAPPVQKVSVVIPVY"
    tokens = tokenize_sequence(seq, max_len=64)
    decoded = decode_sequence(tokens)
    assert seq == decoded, f"Roundtrip failed! Expected {seq}, got {decoded}"

def test_tokenizer_with_unknown():
    """Test tokenizer handles unknown amino acids gracefully"""
    seq = "MFSZAPP"  # 'Z' is not in standard 20 AA vocab
    tokens = tokenize_sequence(seq, max_len=32)
    decoded = decode_sequence(tokens)

    # Expect roundtrip mismatch (since Z → <UNK>)
    assert decoded != seq
    assert "?" in decoded or "<UNK>" not in vocab, "Unknown AA not handled correctly"
    assert len(tokens) == 32, "Token length mismatch after padding"