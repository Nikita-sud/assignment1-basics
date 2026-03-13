import regex as re
from collections import Counter

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {}
    special_token_ids = {}
    for token in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = token.encode("utf-8")
        special_token_ids[token] = token_id

    byte_token_ids = {}
    for byte in range(256):
        token_id = len(vocab)
        vocab[token_id] = bytes([byte])
        byte_token_ids[byte] = token_id

    merges = []
    word_counts = Counter()

    def add_text_tokens(text_chunk: str) -> None:
        for match in re.finditer(GPT2_PATTERN, text_chunk):
            word = match.group()
            word_tuple = tuple(byte_token_ids[byte] for byte in word.encode("utf-8"))
            word_counts[word_tuple] += 1

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
        start = 0
        for match in special_pattern.finditer(text):
            add_text_tokens(text[start:match.start()])
            word_counts[(special_token_ids[match.group()],)] += 1
            start = match.end()
        add_text_tokens(text[start:])
    else:
        add_text_tokens(text)

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        pair_counts = Counter()
        for word_tuple, count in word_counts.items():
            for pair in zip(word_tuple, word_tuple[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        best_id = len(vocab)
        vocab[best_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_word_counts = Counter()
        for word_tuple, count in word_counts.items():
            new_word_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                    new_word_tuple.append(best_id)
                    i += 2
                else:
                    new_word_tuple.append(word_tuple[i])
                    i += 1
            new_word_counts[tuple(new_word_tuple)] += count
        word_counts = new_word_counts

    return vocab, merges