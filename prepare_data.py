"""Tokenize TinyStories .txt files into .bin files for training."""
import numpy as np
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

TRAIN_TXT = "data/TinyStoriesV2-GPT4-train.txt"
VAL_TXT = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_BIN = "data/tinystories_train.bin"
VAL_BIN = "data/tinystories_val.bin"


def main():
    print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE} on {TRAIN_TXT}...")
    vocab, merges = train_bpe(TRAIN_TXT, VOCAB_SIZE, SPECIAL_TOKENS)
    print(f"Done. Vocab size: {len(vocab)}, Merges: {len(merges)}")

    tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)

    for txt_path, bin_path in [(TRAIN_TXT, TRAIN_BIN), (VAL_TXT, VAL_BIN)]:
        print(f"Tokenizing {txt_path} -> {bin_path}...")
        with open(txt_path, "r", encoding="utf-8") as f:
            token_ids = list(tokenizer.encode_iterable(f))
        arr = np.array(token_ids, dtype=np.uint16)
        arr.tofile(bin_path)
        print(f"  Wrote {len(arr)} tokens to {bin_path}")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
