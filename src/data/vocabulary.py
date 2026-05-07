# This file answers one question: how do we convert between words and numbers?
#  The model can't work with strings. Every word needs to be an integer index. This file handles that mapping in both directions.

import re
import json
from collections import Counter

# strips punctuation and lowercases, keeps vocab clean without a heavy NLP library
# Given "A dog sitting on a mat!" it produces ["a", "dog", "sitting", "on", "a", "mat"].
# 
def tokenize(caption: str) -> list[str]:
    # lower() -> so "Dog" and "dog" are treated as the same word, not two different tokens
    # strip() -> remove leading/trailing whitespace
    
    caption = caption.lower().strip() 

    # The pattern [^a-z0-9\s] means "delete anything that is NOT a letter, digit, or whitespace". This removes commas, periods, quotes, etc.
    caption = re.sub(r"[^a-z0-9\s]", "", caption) 

    # split() -> splits the string into a list of words based on whitespace
    return caption.split()


class Vocabulary:

    # index 0 is reserved for padding, which is used to make all sequences the same length (we don't want uneven rows)
    # index 1 is reserved for the start token - Signals "begin generating now" 
    # index 2 is reserved for the end token - Signals "stop generating now"
    # index 3 is reserved for unknown tokens - Signals "word not in vocabulary"
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        self.word2idx: dict[str, int] = {
            "<pad>": self.PAD_IDX,
            "<start>": self.START_IDX,
            "<end>": self.END_IDX,
            "<unk>": self.UNK_IDX,
        }
        self.idx2word: dict[int, str] = {v: k for k, v in self.word2idx.items()}

    # min_freq: Only add words that appear at least this many times. 
    # This prevents the model from wasting memory on rare words (e.g., "pterodactyl") that won't help it learn the core structure.
    # Words that appear fewer than 5 times are likely typos, proper nouns, or noise. Including them bloats the vocabulary 
    # and forces the model to learn embeddings for tokens it almost never sees — wasted capacity.
    def build(self, captions: list[str], min_freq: int = 5) -> None:
        counter: Counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        # We iterate through the counted words. If a word appears enough times (>= min_freq), we assign it a permanent index.
        for word, freq in counter.items():
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    # Convert a sentence into a list of integers
    # "a dog on a mat" → [1, 47, 312, 89, 47, 501, 2] (Assuming these are the assigned IDs for the words)
    
    def encode(self, caption: str) -> list[int]:
        tokens = tokenize(caption)
        return (
            [self.START_IDX] # Appends START token at the beginning
            + [self.word2idx.get(t, self.UNK_IDX) for t in tokens] # Replaces each word with its index, or UNK if not found
            
            + [self.END_IDX] # Appends END token at the end
        )

    # Convert a list of integers back into a sentence
    # [1, 47, 312, 89, 47, 501, 2] → "a dog on a mat"
    
    def decode(self, indices: list[int]) -> str:
        words = []
        for idx in indices:
            # Stop when we hit the end token
            if idx == self.END_IDX:
                break
            if idx not in (self.START_IDX, self.PAD_IDX): # Skip start and padding tokens
                words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    # save the vocabulary to a file
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)

    # load the vocabulary from a file
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            vocab.word2idx = {k: int(v) for k, v in json.load(f).items()}
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab

    # return the size of the vocabulary
    def __len__(self) -> int:
        return len(self.word2idx)
