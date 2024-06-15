# Approach:
#   - For Training (learning token vocabulary)
#   - For Inference (using vocabulary to encode into tokens greedily), use Trie. Advanced: Support token suggestions?

# Step 0: Initialize vocabulary with 256 tokens, each corresponding to a byte.
# Step 1: Pre-tokenize based on regex.
# Step 2 (Initialize, requires going through corpus once O(N = number of tokens in training set)): For each pre-token, 
#   - TokenPairs: create a Tuple of TransientTokenPair (a tuple / named-tuple with (int token-idx1, int token-idx2, bool removed))
#   - TokenPairMaxHeap: Create a heapq of tuples where the first element is the key for comparison (count), and the second element is the custom object, which has the token-id pair, and the pre-token + location within the pre-token)
#
# Step 3: 
#   - Pop max TokenPair from TokenPairMaxHeap, add to vocabulary
#   - For each pre-token, for each location with max TokenPair, --> amortized O(? -- something about overall compression ratio)
#       - mark that TransientTokenPair as removed
#       - find the adjacent TransientTokenPair that aren't removed, 
#           - decrement the counts for these pairs in the Heap (!!! NOT POSSIBLE using a standard heap), 
#           - change the TransientTokenPairs to reflect this new token, increment the new TokenPair counts and add them back to the heap
# Required data-strucutres:
#           - At every step of BPE, we can maintain an add_dict, and a remove_dict:
#                   - The add_dict comprises of new token pairs and their counts.
#                   - The remove_dict comprises of TokenPair objects that are no longer valid
#                   - Modification can be achieved by marking adjacent token-pairs inside remove_dict, and also adding them to add_dict, but with a decreased count.
# For each pre-token:
# Compression Ratio v/s # Tokens used trade-off ():
#   - Per sequence, you can compute # unicode chars in sequence / # tokens in sequence 
#   - ? Find trailing subsequences having a very bad compression ratio

import regex as re
import heapq
from typing import Dict, Tuple, List
from collections import defaultdict

test_string = """ low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# list(pretokens_utf8_seq[0]) would be the list of integers. At this point, each pretoken is in the |token_vocab|

TokenPair = Tuple[int, int]

# Index in pretoken --> (Token Pair)
class Utf8PreTokenTokenPairs:
    # idx at which a TokenPair exists in the PreToken.
    token_pairs: Dict[int, TokenPair]
    invalid_set: set
    def __init__(self, pretoken_str: str):
        pretoken_utf8_seq = pretoken_str.encode("utf-8")
        self.token_pairs = {idx: (pretoken_utf8_seq[idx], pretoken_utf8_seq[idx + 1]) for idx in range(len(pretoken_utf8_seq) - 1)}
        self.invalid_idx = set()
    
    def get_prev_valid(self, seed_idx: int):
        idx = seed_idx - 1
        while idx >= 0:
            if idx in self.invalid_set:
                idx = idx - 1
            return (idx, self.token_pairs[idx])
        return None

    def get_next_valid(self, seed_idx: int):
        idx = seed_idx + 1
        while idx < len(self.token_pairs):
            if idx in self.invalid_set:
                idx = idx + 1
            return (idx, self.token_pairs[idx])
        return None
        
    def token_length(self):
        return len(self.token_pairs) - len(self.invalid_set)

TokenizationCorpus = List[Utf8PreTokenTokenPairs]
# Pretoken ID --> List of locations in the pre-token
PretokenLocations = Dict[int, List[int]]

class TokenPairCorpusMap:
    # Initialize from |self.pretoken_token_pairs|
    def __init__(self):
        self.token_pair_corpus_info: Dict[TokenPair, PretokenLocations] = defaultdict(lambda: defaultdict(list))
    
    def process_corpus(self, corpus: TokenizationCorpus):
        for pretoken_idx, pretoken_token_pairs in enumerate(corpus):
            for token_pair, location_idx in pretoken_token_pairs.token_pairs.items():
                # Since this function should be called before training, none of the locations should be invalid
                assert(location_idx not in pretoken_token_pairs.invalid_set) 
                self.add_pretoken_location_from_corpus_map(token_pair, pretoken_idx, location_idx)

    def add_pretoken_location_from_corpus_map(self, token_pair: TokenPair, pretoken_idx: int, location_idx: int):
        self.token_pair_corpus_info[token_pair][pretoken_idx].append(location_idx)
    
    def remove_pretoken_location_from_corpus_map(self, token_pair: TokenPair, pretoken_idx: int, location_idx: int):
        if token_pair in self.token_pair_corpus_info:
            if location_idx in self.token_pair_corpus_info[token_pair]:
                if location_idx in self.token_pair_corpus_info[token_pair][pretoken_idx]:
                    self.token_pair_corpus_info[token_pair][pretoken_idx].remove(location_idx)
                    # If the list for location_key is empty, remove the pretoken_idx
                    if not self.token_pair_corpus_info[token_pair][pretoken_idx]:
                        del self.token_pair_corpus_info[token_pair][pretoken_idx]
                    # If the inner dictionary for token_pair is empty, remove the token_pair
                    if not self.token_pair_corpus_info[token_pair]:
                        del self.token_pair_corpus_info[token_pair]

    def get_token_pair_count(self, token_pair: TokenPair):
        return sum(len(locations) for locations in self.token_pair_corpus_info[token_pair].values())


# Index in pretoken --> (Token Pair)
# TODO: Heapq forces updates via add and remove. Consider having a custom version since this can be inefficient.
class TokenPairUpdatableMaxHeap:
    """
    TokenPairMaxHeap: An updatable heap of TokenPairs, where priority is their net count.
        - Support accumulating new Token pairs and adding them to the heap (e.g. upon initialization or adding a new Token to the vocabulary)
        - Support finding a specific Token pair in the heap and changing count. This can be accomplished by a Dictionary that allows marking certain elements of the heap as "REMOVED", decrementing is

    To add:
       - 
    To remove:
       - Exchange with last element
       - Remove last element
       - Call heapify

    # Implement using a) Heap algorithm + b) Dict algorithm and compare efficiency in training.
    """

    def __init__(pretoken_byte_pairs: TokenizationCorpus):
        """
        Initializes the heap with count (int) as the priority
        """    
        priority_heap: List[Tuple(int, TokenPairCorpusInfo)]
        staleset()
        # Convert .token_pairs to a count dict

    def update_or_add(self, token_pair_info: TokenPairCorpusInfo):
        # 
        if token_pair_info.token_pair in 

    def remove(self, remove_list: List[TokenPair]):
        # Change with 
        self.
    def insert_heap(self, add_dict: Dict[TokenPair, int]):
    def pop_max(remove_list: List[Tuple[int, int]]):
    
    
class BPETokenizer:
    def __init__(self, text_corpus: str):
        # Initilize Utf8PreTokenBytePairs. This is a self-contained representation of the corpus.
        # NOTE: As the token-vocabulary expands during training, some of these idx will become stale (will be marked invalid)
        self.training_corpus = TokenizationCorpus([Utf8PreTokenTokenPairs(pretoken) for pretoken in re.findall(PRETOKEN_PATTERN, text_corpus)])
        self.token_pair_corpus_map = TokenPairCorpusMap()
        self.token_pair_corpus_map.process_corpus(self.training_corpus)

        # Initialize TokenPairMaxHeap.
        self.token_pair_max_heap = TokenPairUpdatableMaxHeap(self, self.token_pair_corpus_map)
        self.token_vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self):
        num_merges = 1
        for i in range(num_merges):
            self.train_one_step()

    # TODO: Consider parallelizing the creation of |token_pair_add_dict| and |token_pair_remove_dict| and serializing the udpate
    def train_one_step(self):
        # Pop and extract TokenPairCorpusInfo
        max_token_pair_info = self.token_pair_max_heap.pop_max()[1]

        # Create a new token, concatenating bytes from associated existing tokens
        self.token_vocab[len(self.token_vocab)] = map(max_token_pair_info.token_pair, self.token_vocab.get)
        new_token = len(self.token_vocab)
        
        # Update |token_pair_corpus_map|: Find all affected token-pairs adjacent to this token in the pretoken corpus
        # Maintain a set of all token-pairs whose counts will change to adjust max-heap
        changed_token_pairs = set(TokenPair)
        for pretoken_idx, locations in max_token_pair_info.pretoken_locations.items():
            for location in locations:
                # Invalidate the current location
                self.pretoken_token_pairs[pretoken_idx].set_invalid(location)
                # Find adjacent tokens: TODO: Use soft-deletion to invalidate previous token-pairs that are now merged into a single token.
                next_loc, next_token_pair = self.pretoken_token_pairs[pretoken_idx].get_next_valid(location)
                prev_loc, prev_token_pair = self.pretoken_token_pairs[pretoken_idx].get_previous_valid(location)

                # Update corpus map: Create new token pairs for next token pair
                new_token_pair_next = (new_token, next_token_pair[1])
                self.corpus_map.remove_pretoken_location_from_corpus_map(next_token_pair, pretoken_idx, next_loc)
                self.corpus_map.add_pretoken_location_from_corpus_map(new_token_pair_next, pretoken_idx, next_loc)
                new_token_pair_prev = (prev_token_pair[0], new_token)
                self.corpus_map.remove_pretoken_location_from_corpus_map(next_token_pair, pretoken_idx, prev_loc)
                self.corpus_map.add_pretoken_location_from_corpus_map(new_token_pair_prev, pretoken_idx, prev_loc)

                # Add changed token-pairs to change-list
                changed_token_pairs.add(next_token_pair)
                changed_token_pairs.add(prev_token_pair)
                changed_token_pairs.add(new_token_pair_next)
                changed_token_pairs.add(new_token_pair_prev)


        # Update |token_pair_max_heap| based on |changed_token_pairs|

    def encode(self):
        """
        Uses a Trie for efficient lookups of incoming strings.
        """
        raise NotImplementedError
    
    def decode(self):
        raise NotImplementedError