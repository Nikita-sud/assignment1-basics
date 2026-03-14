import regex as re
from collections import Counter

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {}
    special_token_ids = {}

    merges = []
    word_counts = Counter()
    return vocab, merges