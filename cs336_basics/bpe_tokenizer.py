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
# from .bpe_corpus_graph import BpeCorpusGraph, TokenPair, create_corpus_graph_from_file
from dataclasses import dataclass
import pickle
import shelve
import pathlib

TokenPair = Tuple[int, int]

PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
compiled_pattern = re.compile(PRETOKEN_PATTERN)

# list(pretokens_utf8_seq[0]) would be the list of integers. At this point, each pretoken is in the |token_vocab|

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

    def __init__(self, pretoken_str: str):
        pretoken_utf8_seq = pretoken_str.encode("utf-8")
        self.token_pairs = {
            idx: (pretoken_utf8_seq[idx], pretoken_utf8_seq[idx + 1])
            for idx in range(len(pretoken_utf8_seq) - 1)
        }
        self.invalid_idx_set = set()

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


# {Pretoken-ID: [List of locations in the pre-token]}
PretokenLocations = Dict[int, List[int]]

from collections import defaultdict
import ast

class TokenPairCorpusMap:
    # Initialize from |self.pretoken_token_pairs|
    def __init__(self, shelf_path: str):
        # Maintain token-pair counts in RAM
        self.token_counts = {}

        # Maintain Dict[TokenPair, PretokenLocations] in shelf
        self.shelf_path = shelf_path

    def get_pretoken_locations(self, token_pair: TokenPair):
        with shelve.open(self.shelf_path, writeback=True) as token_pair_corpus_map_shelf:
            return token_pair_corpus_map_shelf[str(token_pair)]

    def process_corpus(self, corpus_shelf_path: str):
        with shelve.open(corpus_shelf_path) as corpus_shelf, shelve.open(self.shelf_path, writeback=True) as token_pair_corpus_map_shelf:
            for pretoken_idx_str, pretoken_token_pairs in corpus_shelf.items():
                pretoken_idx = int(pretoken_idx_str)
                for location_idx, token_pair in pretoken_token_pairs.token_pairs.items():
                    token_pair_str = str(token_pair)
                    # Since this function should be called before training, none of the locations should be invalid
                    assert location_idx not in pretoken_token_pairs.invalid_idx_set
                    # Check if the key exists in the shelf
                    if token_pair_str not in token_pair_corpus_map_shelf:
                        token_pair_corpus_map_shelf[token_pair_str] = {pretoken_idx: [location_idx]}
                    elif pretoken_idx not in token_pair_corpus_map_shelf[token_pair_str]:
                        token_pair_corpus_map_shelf[token_pair_str][pretoken_idx] = [location_idx]
                    else:
                        token_pair_corpus_map_shelf[token_pair_str][pretoken_idx].append(location_idx)
           
        
        # Token counts:
        with shelve.open(self.shelf_path) as token_pair_corpus_map_shelf:
            for token_pair_str, pretoken_locs in token_pair_corpus_map_shelf.items():
                total_count = sum([len(locations) for locations in pretoken_locs.values()])
                token_pair = ast.literal_eval(token_pair_str)
                self.token_counts[token_pair] = total_count
        
        print(self.token_counts)
    

    def change_corpus(self, info_add: Dict[TokenPair, PretokenLocations], info_remove: Dict[TokenPair, PretokenLocations]):
        with shelve.open(self.shelf_path, writeback=True) as token_pair_corpus_map_shelf:
            # Add items
            for token_pair, pretoken_locs in info_add.items():
                token_pair_str = str(token_pair)
                if token_pair_str in token_pair_corpus_map_shelf.keys():
                    for pretoken_idx, locations in pretoken_locs.items():
                        if pretoken_idx in token_pair_corpus_map_shelf[token_pair]:
                            token_pair_corpus_map_shelf[token_pair_str][pretoken_idx] += locations
                        else:
                            token_pair_corpus_map_shelf[token_pair_str][pretoken_idx] = info_add[token_pair][pretoken_idx]
                else:
                    token_pair_corpus_map_shelf[token_pair_str] = info_add[token_pair]

            # Remove items; assume that all info_remove are present
            for token_pair, pretoken_locs in info_remove.items():
                token_pair_str = str(token_pair)
                for pretoken_idx, locations in pretoken_locs.items():
                    token_pair_corpus_map_shelf[token_pair_str][pretoken_idx] = [loc for loc in token_pair_corpus_map_shelf[token_pair_str][pretoken_idx] if loc not in set(locations)]
                    # for location_idx in locations:
                    #     token_pair_corpus_map_shelf[token_pair_str][pretoken_idx].remove(
                    #         location_idx
                    #     )
                    # If the list for pretoken_idx is empty, remove the pretoken_idx
                    if not token_pair_corpus_map_shelf[token_pair_str][pretoken_idx]:
                        del token_pair_corpus_map_shelf[token_pair_str][pretoken_idx]
                    # If the inner dictionary for token_pair is empty, remove the token_pair
                    if not token_pair_corpus_map_shelf[token_pair_str]:
                        del token_pair_corpus_map_shelf[token_pair_str]
        
        # Modify counts
        for token_pair, pretoken_locs in info_add.items():
            self.token_counts[token_pair] += len(pretoken_locs)

        for token_pair, pretoke_locs in info_remove.items():
            self.token_counts[token_pair] -= len(pretoke_locs)
    
    def get_token_pair_count(self, token_pair: TokenPair):
        return self.token_counts[token_pair]


# TokenTrie for efficient encoding of an arbitrary byte-string given a token vocabulary after training
class ByteTrieNode:
    # Ensure that each instance gets a separate dict.
    def __init__(self):
        # A dict {next-byte: ByteTrieNode}
        self.children: dict = {}
        self.token: int = None


class TokenTrieEncoderDecoder:
    def __init__(self, token_vocab: Dict[int, bytes]):
        self._root = ByteTrieNode()
        self.valid = False
        self.token_vocab = token_vocab

    def build(self):
        for token, byte_string in self.token_vocab.items():
            node = self._root
            # Keep adding nodes if they don't exist already.
            # At end of the loop, node->root should represent the token byte-string
            for b in byte_string:
                # TODO: Might be able to use a default-dict
                if b not in node.children:
                    node.children[b] = ByteTrieNode()
                node = node.children[b]
            node.token = token

        self.valid = self._validate()

    def _validate(self):
        self._validate_children(self._root)

    def _validate_children(self, start_node: ByteTrieNode):
        """
        Due to the nature of the vocabulary, each child-node in the Trie must have
        """
        for b, child_node in start_node.children.items():
            assert child_node.token is not None, (
                "ERROR: " + str(b) + ": Does not have an associated token-id."
            )
            self._validate_children(child_node)

    def tokenize(self, input_byte_str: bytes) -> List[int]:
        # Convert the immutable byte string to a mutable bytearray
        byte_array = bytearray(input_byte_str)
        node = self._root
        tokenized_str = []
        while byte_array:
            # Pop the first byte
            first_byte = byte_array.pop(0)
            if first_byte in node.children:
                node = node.children[first_byte]
            else:
                # No more matching possible. Return token associated and process remaining string from root
                assert node.token is not None
                tokenized_str.append(node.token)
                node = self._root
        return tokenized_str

    def encode(self, text: str):
        """
        Given a mapping of int --> byte-strings,
        Uses a Trie for efficient lookups of incoming byte-strings to greedily .
        """
        assert self.valid, "Error: Trie is invalid."
        return self.tokenize(text.encode("utf-8"))


# We store Token and counts in a custom heap to control efficiency during modificaitons
class MyBPETokenizer:
    # Use if True, uses heap to find max-freq token-pair
    # if False, computes maximal element in token_pair_corpus_map directly
    PRETOKEN_SHELF_PATH = '/tmp/pretoken_shelf.db'
    TOKEN_PAIR_CORPUS_MAP_SHELF_PATH = '/tmp/token_pair_corpus_map.db'

    def __init__(self, text_corpus: str, special_tokens: List[str] = []):
        # Initilize Utf8PreTokenBytePairs. This is a self-contained representation of the corpus.
        # NOTE: As the token-vocabulary expands during training, some of these idx will become stale (will be marked invalid)
        with shelve.open(self.PRETOKEN_SHELF_PATH) as shelf:
            for idx, pretoken in enumerate(re.findall(PRETOKEN_PATTERN, text_corpus)):
                shelf[str(idx)] = Utf8PreTokenTokenPairs(pretoken)

        # Initialize special tokens before all others
        self.token_vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Initialize Token book-keeping (both dict and heap data-structures)
        self.token_pair_corpus_map = TokenPairCorpusMap(self.TOKEN_PAIR_CORPUS_MAP_SHELF_PATH)
        self.token_pair_corpus_map.process_corpus(self.PRETOKEN_SHELF_PATH)
        breakpoint()
        self.token_pair_priority_queue = ModifiablePriorityQueue.heapify(
            [
                HeapItem(token_pair, (self.token_pair_corpus_map.get_token_pair_count(token_pair), 
                                      tuple(map(self.token_vocab.get, token_pair))))
                for token_pair in self.token_pair_corpus_map.token_counts.keys()
            ]
        )

        # Special tokens are only added to the vocabulary after training
        self.special_tokens = special_tokens

        # merges: list[tuple[bytes, bytes]]
        #      BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
        #      representing that <token1> was merged with <token2>.
        #      Merges are ordered by order of creation.
        self.merges = []

    def get_merges(self) -> List[Tuple]:
        return self.merges

    def add_special_tokens_to_vocab(self):
        # Add special tokens
        len_vocab_no_special = len(self.token_vocab)
        for idx, sp_token in enumerate(self.special_tokens):
            self.token_vocab[len_vocab_no_special + idx] = sp_token.encode('utf-8')  # No need to encode

    def get_vocab(self) -> Dict[int, bytes]:
        # vocab: dict[int, bytes]
        #       The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
        #       to bytes (token bytes)
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
                self.token_pair_priority_queue.add_task(token_pair, (count, utf8_byte_str_pair))

    def train(self, num_merges: int):
        for i in range(num_merges):
            if i % 100 == 0:
                print(f'> {i} merges complete')
            self.train_one_step()

    def train_one_step(self):
        # Pop and extract TokenPairCorpusInfo
        top_heap_item = self.token_pair_priority_queue.pop_task()
        chosen_token_pair, freq = (top_heap_item.name, top_heap_item.priority[0])

        # print(f"Merge #{len(self.merges) + 1} {token_seq_to_bytes(chosen_token_pair, self.token_vocab)} with freq: {freq}")

        chosen_token_pair_pretoken_locations = self.token_pair_corpus_map.get_pretoken_locations(chosen_token_pair)
        

        # Create a new token (increment token-vocab size) representing the token_pair, concatenating bytes from associated existing tokens
        new_token = len(self.token_vocab)
        new_merged_bytestring_pair = tuple(map(self.token_vocab.get, chosen_token_pair))
        self.merges.append(new_merged_bytestring_pair)
        self.token_vocab[new_token] = b"".join(new_merged_bytestring_pair)

        # Update |token_pair_corpus_map|: Find all affected token-pairs adjacent to this token in the pretoken corpus
        # Maintain a set of all token-pairs whose counts will change to adjust max-heap
        changed_token_pairs = set()
        token_pair_add_info: Dict[TokenPair, PretokenLocations] = defaultdict(
            lambda: defaultdict(list)
        )
        token_pair_remove_info: Dict[TokenPair, PretokenLocations] = defaultdict(
            lambda: defaultdict(list)
        )
        token_pair_remove_info[chosen_token_pair] = chosen_token_pair_pretoken_locations

        # TODO: Read relevant self.training_corpus[pretoken_idx] from shelf and cache here --> Read-only
        #       Do one pass of this loop and gather all the token pairs that would change, and extract from self.token_pair_corpus_map
        with shelve.open(self.PRETOKEN_SHELF_PATH, writeback=True) as pretokens_shelf:
            for pretoken_idx, locations in chosen_token_pair_pretoken_locations.items():
                # Open 
                
                for location in locations:
                    # Invalidate the current location
                    pretokens_shelf[str(pretoken_idx)].set_invalid(location)

                    # Find adjacent tokens:
                    adj_next = pretokens_shelf[str(pretoken_idx)].get_next_valid(
                        location
                    )  # can return None
                    if adj_next is not None:
                        # Update corpus map: Create new token pairs for next token pair
                        next_loc, next_token_pair = adj_next
                        new_token_pair_next = (new_token, next_token_pair[1])
                        token_pair_remove_info[next_token_pair][pretoken_idx].append(next_loc)
                        token_pair_add_info[new_token_pair_next][pretoken_idx].append(next_loc)
                    
                        # Update corpus
                        pretokens_shelf[str(pretoken_idx)].token_pairs[next_loc] = new_token_pair_next

                        # Add changed token-pairs to change-list
                        changed_token_pairs.add(next_token_pair)
                        changed_token_pairs.add(new_token_pair_next)

                    adj_prev = pretokens_shelf[pretoken_idx].get_prev_valid(
                        location
                    )  # can return None
                    if adj_prev is not None:
                        # Update corpus map: Create new token pairs for next token pair
                        prev_loc, prev_token_pair = adj_prev
                        new_token_pair_prev = (prev_token_pair[0], new_token)
                        token_pair_remove_info[prev_token_pair][pretoken_idx].append(prev_loc)
                        token_pair_add_info[new_token_pair_prev][pretoken_idx].append(prev_loc)
                        
                        # Update corpus
                        pretokens_shelf[str(pretoken_idx)].token_pairs[prev_loc] = new_token_pair_prev

                        # Add changed token-pairs to change-list
                        changed_token_pairs.add(prev_token_pair)
                        changed_token_pairs.add(new_token_pair_prev)
        
        # Update Corpus Map shelf
        self.token_pair_corpus_map.change_corpus(token_pair_add_info, token_pair_remove_info)

        # !! Trying Alternative:  Remove from dict
        # del self.token_pair_corpus_map.token_pair_corpus_info[chosen_token_pair]
        # debug_list = [b'ce', b'le', b'@-@', b' are']
        # for changed_tk in changed_token_pairs:
        #     bytes_changed = token_seq_to_bytes(changed_tk, self.token_vocab)
        #     if bytes_changed in debug_list:
        #         print(
        #             f"Change --> {bytes_changed}: {self.token_pair_corpus_map.get_token_pair_count(changed_tk)} occurrances"
        #         ) 

        # Update priority queue
        updated_token_pair_counts = {
            token_pair: self.token_pair_corpus_map.get_token_pair_count(token_pair)
            for token_pair in changed_token_pairs
        }
        self._update_token_pair_priorities(updated_token_pair_counts)


def train_bpe(input_str: str, vocab_size: int, special_tokens: List[str] = []) -> Tuple[Dict[int, bytes], List[Tuple]]: 
    """
    Return: A tuple, where:
    - the first element is the vocabulary, represented as a mapping of token -> bytestring of
    - the second element is an ordered list of merged byte-strings. 
    """
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

# Training on corpus
import pathlib
import json
DATASET_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

def train_bpe_tinystories():
    input_path = DATASET_PATH / "TinyStoriesV2-GPT4-train.txt"
    serialized_vocab_path = DATASET_PATH / "TinyStoriesV2-GPT4_tokenized_vocab.json"
    serialized_merges_path = DATASET_PATH / "TinyStoriesV2-GPT4_tokenized_merges.txt"
    vocab, merges = train_bpe_from_file(
        input_path=input_path,
        vocab_size=100,
        special_tokens=["<|endoftext|>"],
    )
    print("***** Finished training vocab! ******")
    
    # Serialize the vocabulary
    with open(serialized_vocab_path, "w") as file:
        json.dump(vocab, file, indent=4)

    with open(serialized_merges_path, 'w') as file:
        for item in merges:
            file.write(item + '\n')
    
def train_bpe_tinystories():
    input_path = DATASET_PATH / "owt_train.txt"
    serialized_vocab_path = DATASET_PATH / "owt_train_tokenized_vocab.json"
    serialized_merges_path = DATASET_PATH / "owt_train_tokenized_merges.txt"
    vocab, merges = train_bpe_from_file(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    
    # Serialize the vocabulary
    with open(serialized_vocab_path, "w") as file:
        json.dump(vocab, file, indent=4)

    with open(serialized_merges_path, 'w') as file:
        for item in merges:
            file.write(item + '\n')
    

# python -m cs336_basics.bpe_tokenizer

if __name__ == "__main__":

    test_string = """ low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    vocab, merges = train_bpe(test_string, 264)
    print(f"Vocab: {vocab}\n------\n")
    print(f"Merges: {merges}")

    # train_bpe_tinystories()
