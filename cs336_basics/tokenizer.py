import json
from collections.abc import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in cls._gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab = json.load(vocab_f)

        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merge_token_1, merge_token_2 = cleaned_line.split(" ")
                    merges.append(
                        (
                            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                        )
                    )

        if special_tokens:
            vocab_values = set(vocab.values())
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab_values:
                    vocab[len(vocab)] = special_token_bytes
                    vocab_values.add(special_token_bytes)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    def encode(self, text: str) -> list[int]:
        final_token_ids = []
        special_token_set = set(self.special_tokens)

        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            if part in special_token_set:
                final_token_ids.append(self.byte_to_id[part.encode("utf-8")])
                continue

            for match in re.finditer(self.GPT2_PATTERN, part):
                token_bytes = match.group().encode("utf-8")
                word = tuple(bytes([b]) for b in token_bytes)

                while len(word) > 1:
                    pairs = set(zip(word, word[1:]))
                    best_pair = min(
                        (pair for pair in pairs if pair in self.merge_priority),
                        key=lambda pair: self.merge_priority[pair],
                        default=None,
                    )
                    if best_pair is None:
                        break
                    word = self.merge_word(word, best_pair, best_pair[0] + best_pair[1])

                for token in word:
                    final_token_ids.append(self.byte_to_id[token])
                
        return final_token_ids
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    @staticmethod
    def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
        merged_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                merged_word.append(new_token)
                i += 2
            else:
                merged_word.append(word[i])
                i += 1
        return tuple(merged_word)

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b"".join(self.vocab[idx] for idx in ids)
        return byte_sequence.decode("utf-8", errors="replace")
