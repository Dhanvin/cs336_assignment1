# Approach:
#   - For Training (learning token vocabulary),
#       - For each pre-token, uses Utf8PreTokenTokenPairs to keep track of token-pair locations, changed token-pairs, as well as stale token-pairs (due to a token pair being included in library)
#       - Maintains TokenPairCorpusMap, which is a list of all locations where a token-pair is present (uses defaultdict for insertion-convenience)
#       - Uses sortedcontainers.SortedDict to find the max TokenPair at any time. (experimenting)
#   - For Inference (using vocabulary to encode into tokens greedily), use Trie. Advanced: Support token suggestions?
#
# For each pre-token:
# Compression Ratio v/s # Tokens used trade-off ():
#   - Per sequence, you can compute # unicode chars in sequence / # tokens in sequence
#   - ? Find trailing subsequences having a very bad compression ratio

## TODO:
#     -- Tokenizer should accept user-defined tokens provided to the constructor
#     -- While encoding a large file /  data stream, we need to break it up into chunks for constant memory, ensuring that pretokens don't cross chunk boundaries
#     -- Test the tokenizer in test_tokenizer.py and tests/test_train_bpe.py
#     -- Experiment with tokenizers


import regex as re
from typing import Dict, Tuple, List
from .modifiable_priority_queue import ModifiablePriorityQueue, HeapItem
from dataclasses import dataclass

PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
# list(pretokens_utf8_seq[0]) would be the list of integers. At this point, each pretoken is in the |token_vocab|

TokenPair = Tuple[int, int]


def token_seq_to_bytes(
    token_seq: List[int] | TokenPair, token_vocab: Dict[int, bytes]
) -> bytes:
    bytes_list = list(map(token_vocab.get, token_seq))
    return b"".join(bytes_list)


def decode(token_seq: List[int] | TokenPair, token_vocab: Dict[int, bytes]) -> str:
    return token_seq_to_bytes(token_seq, token_vocab).decode("utf-8")


# Index in pretoken --> (Token Pair)
class Utf8PreTokenTokenPairs:
    # idx at which a TokenPair exists in the PreToken.
    token_pairs: Dict[int, TokenPair]
    invalid_idx_set: set
    pretoken_utf8_b: bytes

    def __init__(self, pretoken_str: str, token_vocab: Dict[bytes, int]):
        self.pretoken_utf8_b = pretoken_str.encode("utf-8")
        # print(self.pretoken_utf8_b)
        # TODO: Handle case where bytes in a pretoken are out of vocab ("out-of-distribution" / tokenizer-training process deficiency)
        #   Thoughts - 1) append to vocab and create tokens for them (so decoding preserves them)
        #              2) Simply ignore these bytes.
        #  Currently trying option 2, because after training, we should not be changing the tokenizer state.
        self.token_pairs = {}
        self.invalid_idx_set = set()

        # breakpoint()
        for idx in range(len(self.pretoken_utf8_b) - 1):
            this_byte = self.pretoken_utf8_b[idx : idx + 1]
            next_byte = self.pretoken_utf8_b[idx + 1 : idx + 2]
            if this_byte not in token_vocab or next_byte not in token_vocab:
                self.invalid_idx_set.add(idx)
                continue
            self.token_pairs[idx] = (token_vocab[this_byte], token_vocab[next_byte])

    def set_invalid(self, loc: int):
        self.invalid_idx_set.add(loc)

    def get_prev_valid(self, seed_idx: int):
        idx = seed_idx - 1
        while idx >= 0:
            if idx in self.invalid_idx_set:
                idx = idx - 1
            else:
                return (idx, self.token_pairs[idx])
        return None

    def get_next_valid(self, seed_idx: int):
        idx = seed_idx + 1
        while idx < len(self.token_pairs):
            if idx in self.invalid_idx_set:
                idx = idx + 1
            else:
                return (idx, self.token_pairs[idx])
        return None

    def token_length(self):
        return len(self.token_pairs) - len(self.invalid_idx_set)

    def tokenize(self) -> List[int]:
        out = []
        ordered_token_pairs = [
            self.token_pairs[loc]
            for loc in sorted(self.token_pairs.keys())
            if loc not in self.invalid_idx_set
        ]

        prev = None
        for token_pair in ordered_token_pairs:
            if token_pair[0] == prev:
                out.append(token_pair[1])
            else:
                out.append(token_pair[0])
                out.append(token_pair[1])
            prev = token_pair[1]
        return out


# {Pretoken-ID: [List of locations in the pre-token]}
PretokenLocations = Dict[int, List[int]]

from collections import defaultdict


class MergeResults:
    def __init__(self) -> None:
        # A set of all token-pairs whose counts will change to adjust max-heap
        self.changed_token_pairs = set()
        # Pretokens with singular tokens (if adj_next and adj_prev are None).
        self.lonely_tokens = dict()  # pretoken --> token


class TokenPairCorpusMap:
    # Initialize from |self.pretoken_token_pairs|
    def __init__(self, corpus: Dict[int, Utf8PreTokenTokenPairs]):
        self.token_pair_corpus_info: Dict[TokenPair, PretokenLocations] = defaultdict(
            lambda: defaultdict(list)
        )
        self.corpus = corpus

        for pretoken_idx, pretoken_token_pairs in corpus.items():
            for location_idx, token_pair in pretoken_token_pairs.token_pairs.items():
                # Since this function should be called before training, none of the locations should be invalid
                assert location_idx not in pretoken_token_pairs.invalid_idx_set
                self.add_pretoken_location_to_corpus_map(
                    token_pair, pretoken_idx, location_idx
                )

    def tokenize(self) -> Dict[int, List[int]]:
        return {idx: el.tokenize() for idx, el in self.corpus.items()}

    def add_pretoken_location_to_corpus_map(
        self, token_pair: TokenPair, pretoken_idx: int, location_idx: int
    ):
        self.token_pair_corpus_info[token_pair][pretoken_idx].append(location_idx)

    def remove_pretoken_location_from_corpus_map(
        self, token_pair: TokenPair, pretoken_idx: int, location_idx: int
    ):
        if token_pair in self.token_pair_corpus_info:
            if pretoken_idx in self.token_pair_corpus_info[token_pair]:
                if (
                    location_idx
                    in self.token_pair_corpus_info[token_pair][pretoken_idx]
                ):
                    self.token_pair_corpus_info[token_pair][pretoken_idx].remove(
                        location_idx
                    )
                    # If the list for pretoken_idx is empty, remove the pretoken_idx
                    if not self.token_pair_corpus_info[token_pair][pretoken_idx]:
                        del self.token_pair_corpus_info[token_pair][pretoken_idx]
                    # If the inner dictionary for token_pair is empty, remove the token_pair
                    if not self.token_pair_corpus_info[token_pair]:
                        del self.token_pair_corpus_info[token_pair]

    def get_token_pair_count(self, token_pair: TokenPair):
        if token_pair not in self.token_pair_corpus_info:
            return 0
        return sum(
            len(locations)
            for locations in self.token_pair_corpus_info[token_pair].values()
        )

    def get_all_token_pairs(self):
        return self.token_pair_corpus_info.keys()

    # Returns a set of tokens with changed counts
    def merge_token(self, chosen_token_pair: TokenPair, new_token: int) -> MergeResults:
        chosen_token_pair_pretoken_locations: PretokenLocations = (
            self.token_pair_corpus_info[chosen_token_pair]
        )

        results = MergeResults()
        # NOTE: No need to update token_pair_corpus_map[chosen_token_pair] or
        results.changed_token_pairs.add(chosen_token_pair)
        for pretoken_idx, locations in chosen_token_pair_pretoken_locations.items():
            for location in locations:
                # Invalidate the current location
                self.corpus[pretoken_idx].set_invalid(location)

                # Find adjacent tokens:
                adj_next = self.corpus[pretoken_idx].get_next_valid(
                    location
                )  # can return None
                if adj_next is not None:
                    # Update corpus map: Create new token pairs for next token pair
                    next_loc, next_token_pair = adj_next
                    new_token_pair_next = (new_token, next_token_pair[1])
                    self.remove_pretoken_location_from_corpus_map(
                        next_token_pair, pretoken_idx, next_loc
                    )
                    self.add_pretoken_location_to_corpus_map(
                        new_token_pair_next, pretoken_idx, next_loc
                    )

                    # Update corpus
                    self.corpus[pretoken_idx].token_pairs[
                        next_loc
                    ] = new_token_pair_next

                    # Add changed token-pairs to change-list
                    results.changed_token_pairs.add(next_token_pair)
                    results.changed_token_pairs.add(new_token_pair_next)

                adj_prev = self.corpus[pretoken_idx].get_prev_valid(
                    location
                )  # can return None
                if adj_prev is not None:
                    # Update corpus map: Create new token pairs for next token pair
                    prev_loc, prev_token_pair = adj_prev
                    new_token_pair_prev = (prev_token_pair[0], new_token)
                    self.remove_pretoken_location_from_corpus_map(
                        prev_token_pair, pretoken_idx, prev_loc
                    )
                    self.add_pretoken_location_to_corpus_map(
                        new_token_pair_prev, pretoken_idx, prev_loc
                    )
                    # Update corpus
                    self.corpus[pretoken_idx].token_pairs[
                        prev_loc
                    ] = new_token_pair_prev

                    # Add changed token-pairs to change-list
                    results.changed_token_pairs.add(prev_token_pair)
                    results.changed_token_pairs.add(new_token_pair_prev)

                if adj_next is None and adj_prev is None:
                    results.lonely_tokens[pretoken_idx] = new_token

        # Remove merged token and return
        del self.token_pair_corpus_info[chosen_token_pair]
        return results


# We store Token and counts in a custom heap to control efficiency during modificaitons
## TODO (not urgent) --> Broken since we are using huggingface for training
class MyBPETokenizer:
    # Use if True, uses heap to find max-freq token-pair
    # if False, computes maximal element in token_pair_corpus_map directly
    USE_HEAP = True  # NOTE: Currently doesn't support lexicographic ordering...

    def __init__(self, text_corpus: str, special_tokens: List[str] = []):
        # Initialize special tokens before all others
        self.token_vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Initilize Utf8PreTokenBytePairs. This is a self-contained representation of the corpus.
        # NOTE: As the token-vocabulary expands during training, some of these idx will become stale (will be marked invalid)
        training_corpus: TokenizationCorpus = [
            Utf8PreTokenTokenPairs(pretoken, self.token_vocab)
            for idx, pretoken in enumerate(re.findall(PRETOKEN_PATTERN, text_corpus))
        ]

        # Initialize Token book-keeping (both dict and heap data-structures)
        self.token_pair_corpus_map = TokenPairCorpusMap(training_corpus)
        self.token_pair_priority_queue = ModifiablePriorityQueue.heapify(
            [
                HeapItem(
                    token_pair,
                    (
                        self.token_pair_corpus_map.get_token_pair_count(token_pair),
                        tuple(map(self.token_vocab.get, token_pair)),
                    ),
                )
                for token_pair in self.token_pair_corpus_map.get_all_token_pairs()
            ]
        )

        # Special tokens are only added to the vocabulary after training
        self.special_tokens = special_tokens

        # merges: list[tuple[bytes, bytes]] representing an merged token-pairs ordered list training.
        # The ordering is key for the encoding algorithms
        self.merges = []

    def get_merges(self) -> List[Tuple]:
        return self.merges

    def add_special_tokens_to_vocab(self):
        # Add special tokens
        len_vocab_no_special = len(self.token_vocab)
        for idx, sp_token in enumerate(self.special_tokens):
            self.token_vocab[len_vocab_no_special + idx] = sp_token.encode(
                "utf-8"
            )  # No need to encode

    def get_vocab(self) -> Dict[int, bytes]:
        # vocab: dict[int, bytes]
        # The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
        # to bytes (token bytes)
        return self.token_vocab

    def vocab_size(self) -> int:
        return len(self.token_vocab)

    def _update_token_pair_priorities(self, updated_token_pair_counts):
        for token_pair, count in updated_token_pair_counts.items():
            utf8_byte_str_pair = tuple(map(self.token_vocab.get, token_pair))
            if self.token_pair_priority_queue.contains(token_pair):
                self.token_pair_priority_queue.change_priority(
                    token_pair,
                    (count, utf8_byte_str_pair),
                )
            else:
                self.token_pair_priority_queue.add_task(
                    token_pair, (count, utf8_byte_str_pair)
                )

    def train(self, num_merges: int):
        for i in range(num_merges):
            self.train_one_step()

    def train_one_step(self):
        # Pop and extract TokenPairCorpusInfo
        if self.USE_HEAP:
            top_heap_item = self.token_pair_priority_queue.pop_task()
            chosen_token_pair, freq = (top_heap_item.name, top_heap_item.priority[0])
        else:
            # !! Trying Alternative: You could find max from the dictionary itself (more compute)
            chosen_token_pair = max(
                self.token_pair_corpus_map.token_pair_corpus_info,
                key=lambda k: (
                    self.token_pair_corpus_map.get_token_pair_count(k),
                    tuple(map(self.token_vocab.get, k)),
                ),
            )
            freq = self.token_pair_corpus_map.get_token_pair_count(chosen_token_pair)

        # Create a new token (increment token-vocab size) representing the token_pair, concatenating bytes from associated existing tokens
        new_token = len(self.token_vocab)
        new_merged_bytestring_pair = tuple(map(self.token_vocab.get, chosen_token_pair))
        self.merges.append(new_merged_bytestring_pair)
        self.token_vocab[new_token] = b"".join(new_merged_bytestring_pair)

        # Update map with new token
        changed_token_pairs = self.token_pair_corpus_map.merge_token(
            chosen_token_pair, new_token
        ).changed_token_pairs

        # Update priority queue
        if self.USE_HEAP:
            updated_token_pair_counts = {
                token_pair: self.token_pair_corpus_map.get_token_pair_count(token_pair)
                for token_pair in changed_token_pairs
            }
            self._update_token_pair_priorities(updated_token_pair_counts)


def train_bpe(input_str: str, vocab_size: int, special_tokens: List[str] = []):
    tokenizer = MyBPETokenizer(input_str, special_tokens)
    num_merges = vocab_size - tokenizer.vocab_size() - len(special_tokens)
    tokenizer.train(num_merges)
    tokenizer.add_special_tokens_to_vocab()
    return (tokenizer.get_vocab(), tokenizer.get_merges())


def train_bpe_from_file(
    input_path: str, vocab_size: int, special_tokens: List[str] = []
):
    try:
        with open(input_path, "r") as file:
            file_content = file.read()
        return train_bpe(file_content, vocab_size, special_tokens)

    except FileNotFoundError:
        print(f"The file {input_path} does not exist.")


# python -m cs336_basics.bpe_tokenizer
if __name__ == "__main__":

    test_string = """ low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    vocab, merges = train_bpe(test_string, 264)
    print(f"Vocab: {vocab}\n------\n")
    print(f"Merges: {merges}")
