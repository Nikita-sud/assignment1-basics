import regex as re
from collections import Counter
from collections import defaultdict

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str],) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    vocab = {i: bytes([i]) for i in range(256)}
    words = []
    counts = []
    pair_to_words = defaultdict(set)
    special_token_ids = {}
    merges = []

    for secial_token in special_tokens:
        secial_token_byte = secial_token.encode("utf-8")
        token_id=len(vocab)
        special_token_ids[secial_token_byte] = token_id
        vocab[token_id]= secial_token_byte
    
    with open(input_path, "r", encoding="utf-8") as s:
        full_text = s.read()
        pretokens = Counter()
        if special_tokens:
            pattern = "|".join(re.escape(t) for t in special_tokens)
            chunks = re.split(pattern, full_text)
        else:
            chunks = [full_text]
        for chunk in chunks:
            chunk_pretokens = Counter(tuple(bytes([b]) for b in match.group().encode("utf-8")) for match in re.finditer(GPT2_PATTERN, chunk))
            pretokens += chunk_pretokens
        words = list(pretokens.keys())
        counts = list(pretokens.values())

    pair_counts, pair_to_words = count_pairs(words, counts)

    while len(vocab)<vocab_size:
        if not pair_counts:
            print("Больше нет пар для объединения. Остановка.")
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        for idx in list(pair_to_words[best_pair]):
            word = words[idx]
            count = counts[idx]
            
            positions = []
            for i in range(len(word) - 1):
                if (word[i], word[i+1]) == best_pair:
                    positions.append(i)
            
            for i in positions:
                if i > 0:
                    pair_counts[(word[i-1], word[i])] -= count
                pair_counts[(word[i], word[i+1])] -= count
                if i + 2 < len(word):
                    pair_counts[(word[i+1], word[i+2])] -= count
            
            new_word = merge_word(tuple(word), best_pair, new_token)
            words[idx] = list(new_word)

            for i in range(len(new_word)):
                if new_word[i] == new_token:
                    if i > 0:
                        pair_counts[(new_word[i-1], new_word[i])] += count
                        pair_to_words[(new_word[i-1], new_word[i])].add(idx)
                    if i + 1 < len(new_word):
                        pair_counts[(new_word[i], new_word[i+1])] += count
                        pair_to_words[(new_word[i], new_word[i+1])].add(idx)

            

    return vocab, merges

def count_pairs(words,counts):
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, count, i in zip(words,counts, range(len(words))):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(i)
    return pair_counts, pair_to_words

def merge_word(word:tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
    """
    word: tuple of bytes, e.g. (b'h', b'e', b'l', b'l', b'o')
    pair: tuple of two bytes, e.g. (b'l', b'l')
    new_token: bytes, e.g. b'll'
    returns: new tuple after merge
    """

    merged_word = []
    word_length = len(word)
    i = 0
    while i<word_length:
        if(i<word_length-1 and (word[i],word[i+1])==pair):
            merged_word.append(new_token)
            i+=2
        else:
            merged_word.append(word[i])
            i+=1

    return tuple(merged_word)