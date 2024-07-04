# Implements a BPE encoder and decoder

# Input:
#   merges.txt file, with an frequency-ordered set of merges during training
#   vocab.json file, a mapping from tokens to
from typing import List, Dict, Tuple, Iterable, Iterator
import json
import pathlib
import itertools

from .fast_singlethread_tokenizer import (
    Utf8PreTokenTokenPairs,
    PRETOKEN_PATTERN,
    TokenPair,
    TokenPairCorpusMap,
)
import regex as re


def prioritized_regex_matching(
    input_text: str, regex_list: List[re.Pattern]
) -> List[str]:
    all_matches = []

    # Iterate through the regex patterns in order of priority
    # NOTE: substitute by spaces so that we aren't missing other tokens outside that might regex to
    #       the lowest-priority pattern
    mutating_text = input_text
    for priority, regex in enumerate(regex_list):
        substitute_spaces_at = []
        # Find all non-overlapping matches for the current regex
        for match in regex.finditer(mutating_text):
            all_matches.append((match.start(), match.end(), match.group(), priority))
            substitute_spaces_at.append((match.start(), match.end()))

        all_indices = list(
            itertools.chain.from_iterable(
                range(start, end) for start, end in substitute_spaces_at
            )
        )
        mutating_text_list = list(mutating_text)
        # Replace the characters at all the specified indices with spaces
        # TODO(@dmehta) -- BUG: Consider using another character that is not a space.
        rare_char_str = "𓀀"
        for i in all_indices:
            mutating_text_list[i] = rare_char_str
        mutating_text = "".join(mutating_text_list)

    # Sort matches first by priority (lowest number = highest priority), then by start position
    prioritized_matches = sorted(all_matches, key=lambda x: (x[3], x[0]))

    unique_matches: List[Tuple] = []
    covered_positions = set()

    for start, end, matched_text, _ in prioritized_matches:
        # Check if this match overlaps with any previously covered position
        if any(pos in covered_positions for pos in range(start, end)):
            continue
        # Add to result
        unique_matches.append((start, matched_text))
        covered_positions.update(range(start, end))

    # Sometimes, there may be gaps due to merging of a whitespace with rare_char_str.
    # In this case, we create new pre-tokens from the input string
    # TODO(slinky): Is there a better way?
    all_positions = set(range(len(input_text)))
    pretokenization_gaps = all_positions - covered_positions
    for gap in pretokenization_gaps:
        unique_matches.append((gap, input_text[gap]))

    # Sort by start position
    ordered_matches = sorted(unique_matches, key=lambda x: x[0])
    return [match[1] for match in ordered_matches]


class BpePretrainedTokenizer:
    # self.ordered_merges: List[Tuple[bytes, bytes]]
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges=List[Tuple[bytes, bytes]],
        special_tokens: List[str] = None,
    ):
        self.ordered_merges = merges
        self.vocab = vocab

        # For efficiency while encoding,
        # 1) maintain an inverted map to find the token for specific token pairs.
        # 2) maintain an inverted list from byte-pair representing a potential merge --> merge-order
        self.vocab_bytes_to_int = {v: k for k, v in vocab.items()}
        self.token_pair_merge_ranking: Dict[TokenPair, int] = {
            (
                self.vocab_bytes_to_int[bytes_pair[0]],
                self.vocab_bytes_to_int[bytes_pair[1]],
            ): rank
            for rank, bytes_pair in enumerate(self.ordered_merges)
        }

        # Create a prioritized list of compiled regular expressions to ensure that we are first matching for
        # special characters.
        self.pretoken_regex = re.compile(PRETOKEN_PATTERN)
        # Sort special-tokens by length in descending order.
        sorted_special_tokens = (
            sorted(special_tokens, key=len, reverse=True)
            if special_tokens is not None
            else None
        )
        if sorted_special_tokens:
            self.regex_list = [
                re.compile(rf"{re.escape(word)}") for word in sorted_special_tokens
            ]
            self.regex_list += [re.compile(PRETOKEN_PATTERN)]  # Lowest priority
        else:
            self.regex_list = [re.compile(PRETOKEN_PATTERN)]

        # Assumes that the longest sorted special-token is the end-of-text token.
        self.end_of_text_str = sorted_special_tokens[0]
        self.end_of_text_token = (
            self.vocab_bytes_to_int[self.end_of_text_str.encode("utf-8")]
            if sorted_special_tokens is not None
            else None
        )

    # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ):
        # Construct vocab dict. from JSON file with format {unicode-coded string: int, }
        with open(vocab_filepath, "r") as file:
            json_vocab_dict = json.load(file)
        vocab_utf8 = {v: k.encode("utf-8") for k, v in json_vocab_dict.items()}

        # Extract merge-ordering
        merges = []
        lcount = 0
        with open(merges_filepath, "r") as file:
            for line in file:
                lcount += 1
                # Remove newline character
                if line.startswith("#version") or line == "\n":
                    continue  # ignore

                # Remove ending newline and split from the last space.
                merge_strings = line.rstrip("\n").rsplit(" ", 1)

                # Ignore if (could be a formatting error in the trainer)
                if (
                    merge_strings[0] not in json_vocab_dict
                    or merge_strings[1] not in json_vocab_dict
                ):
                    print(
                        f"WARNING: Skipping Line #{lcount}: {line} -- {merge_strings} not in Vocab"
                    )
                    continue

                assert (
                    len(merge_strings) == 2
                ), f"{merge_strings} is not a pair. Line #{lcount}: {line}"
                merges.append(
                    (merge_strings[0].encode("utf-8"), merge_strings[1].encode("utf-8"))
                )

        return cls(vocab_utf8, merges, special_tokens)

    # Encode an input text into a sequence of token IDs.
    def encode(self, text: str) -> List[int]:
        # Used for producing output
        tokenized_pretokens: Dict[int, List[int]] = {}  # {pretoken-idx: List[tokens]}

        # In order to retain the special strings in an input string through the
        # encoder - decoder round-trip, we:
        # 1) Augment the regex with matching for special so they become separate tokens.
        # 2) Add the special strings to the vocab and skip merge-list based encoding
        #    if the entire pre-token is in the vocab
        # pretokens = self.pretoken_regex.findall(text)
        pretokens = prioritized_regex_matching(text, self.regex_list)

        # Store indices where this edge-case applies; we will not use merge-list algorithm here.
        singular_pretoken_ids = set()
        for pretoken_idx, pretoken_str in enumerate(pretokens):
            pretoken_utf8_seq = pretoken_str.encode("utf-8")
            if pretoken_utf8_seq in self.vocab_bytes_to_int:
                singular_pretoken_ids.add(pretoken_idx)
                tokenized_pretokens[pretoken_idx] = [
                    self.vocab_bytes_to_int[pretoken_utf8_seq]
                ]

        # For efficiency, maintain a searchable ordering
        token_pair_map = TokenPairCorpusMap(
            {
                pretoken_idx: Utf8PreTokenTokenPairs(
                    pretoken_str, self.vocab_bytes_to_int
                )
                for pretoken_idx, pretoken_str in enumerate(pretokens)
                if pretoken_idx not in singular_pretoken_ids
            }
        )

        # Get token pair based on minimum-merge-ordering
        def _safe_token_pair_ranker(pair: TokenPair):
            if pair in self.token_pair_merge_ranking:
                return self.token_pair_merge_ranking[pair]
            else:
                # Return a number greater than all other merge
                return len(self.ordered_merges) + 1

        while True:
            ntoken_pairs = len(token_pair_map.token_pair_corpus_info)
            if ntoken_pairs == 0:
                # Encoding over: No token pairs to process
                break
            selected_token_pair = min(
                token_pair_map.token_pair_corpus_info.keys(),
                key=_safe_token_pair_ranker,
            )
            if selected_token_pair not in self.token_pair_merge_ranking:
                # Encoding over: No more merges possible
                break
            # Find corresponding new token from vocab and update token_pair_map and
            new_token_b = b"".join(map(self.vocab.get, selected_token_pair))
            new_token = self.vocab_bytes_to_int[new_token_b]
            lonely_tokens = token_pair_map.merge_token(
                selected_token_pair, new_token
            ).lonely_tokens
            # Insert any lone
            for pretoken_idx, token in lonely_tokens.items():
                assert token == new_token
                tokenized_pretokens[pretoken_idx] = [new_token]

        # Get ordered tokens from token-pair-corpus.
        for pretoken_idx, token_list in token_pair_map.tokenize().items():
            tokenized_pretokens[pretoken_idx] = token_list

        # Sort in order of pretoken-ids
        encoded_token_list_of_lists = [
            tokenized_pretokens[key] for key in sorted(tokenized_pretokens.keys())
        ]

        return list(itertools.chain.from_iterable(encoded_token_list_of_lists))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Input:
            - an iterable of strings (e.g., a Python file handle),
        Return:
            - A generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        token_buffer: List[int] = []
        cnt = 0
        update_log_freq = 10000
        while True:
            if token_buffer:
                yield token_buffer.pop(0)  # Remove and return in order of insertion
            else:
                try:
                    line = next(
                        iterable
                    )  # If iterable is a file in 'r' mode, this returns a line
                    cnt += 1
                    if cnt % update_log_freq == 0:
                        print(f"Processed {cnt: .3e} lines.")
                except StopIteration:
                    print("Encoding over.")
                    break
                token_buffer += self.encode(line)
                if token_buffer:
                    yield token_buffer.pop(0)

    # Decode a sequence of token IDs into text.
    def decode(self, ids: List[int]) -> str:
        utf8_str = b"".join([self.vocab[token] for token in ids])
        return utf8_str.decode("utf-8", errors="replace")


def tokenize_dataset(data_file: str, tokenized_file: str):
    print(f"Tokenizing {data_file} --> {tokenized_file}")
    all_ids: List[int] = []
    start_t = timeit.default_timer()
    with open(data_file) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    file_size = data_file.stat().st_size
    end_t = timeit.default_timer()
    MBps = file_size / (float(end_t - start_t) * 1024.0 * 1024.0)

    # Output efficiency stats
    print(f"BPE: Bytes per token: {file_size / float(len(all_ids)): .2f}")
    print(f"Throughput ratio: {MBps: .2f} MB/s")
    np.save(tokenized_file, np.array(all_ids, dtype=np.uint16))
    tokenized_file_size = tokenized_file.stat().st_size
    print(f"Absolute compression ratio: {file_size / float(tokenized_file_size): .2f}")


import timeit
import numpy as np
from argparse import ArgumentParser
from cs336_basics.transformer.training import get_batch, SamplingStrategy

# python -m cs336_basics.bpe_tokenizer.encoder_decoder
if __name__ == "__main__":
    OVERWRITE_TOKEN_FILES = False
    dataset_name = "TinyStoriesV2-GPT4"
    DATASET_DIR = (
        (pathlib.Path(__file__).resolve()).parent.parent.parent / "data" / dataset_name
    )
    special_token_list = ["<|endoftext|>"]

    # Create Tokenizer. We should ensure that this has the same set of special characters as during training.
    vocab_path = str(DATASET_DIR / "vocab.json")
    merge_path = str(DATASET_DIR / "merges.txt")
    tokenizer = BpePretrainedTokenizer.from_files(
        vocab_path, merge_path, special_tokens=special_token_list
    )
    assert (
        tokenizer.end_of_text_token is not None
    ), "ERROR: No End-of-Text token found in tokenizer"

    # Encode train and validation files using Tokenizer. Save serialized uint16 numpy array of tokens
    DATA_FILE_TRAIN = DATASET_DIR / "TinyStoriesV2-GPT4-train.txt"
    TOKENIZED_FILE_TRAIN = DATASET_DIR / (DATA_FILE_TRAIN.stem + "-tokens.npy")
    if not TOKENIZED_FILE_TRAIN.exists() or OVERWRITE_TOKEN_FILES:
        tokenize_dataset(DATA_FILE_TRAIN, TOKENIZED_FILE_TRAIN)
    else:
        print(f"Skipping: {TOKENIZED_FILE_TRAIN} already exists")

    DATA_FILE_VALIDATION = DATASET_DIR / "TinyStoriesV2-GPT4-valid.txt"
    TOKENIZED_FILE_VALIDATION = DATASET_DIR / (
        DATA_FILE_VALIDATION.stem + "-tokens.npy"
    )
    if not TOKENIZED_FILE_VALIDATION.exists() or OVERWRITE_TOKEN_FILES:
        tokenize_dataset(DATA_FILE_VALIDATION, TOKENIZED_FILE_VALIDATION)
    else:
        print(f"Skipping: {TOKENIZED_FILE_VALIDATION} already exists")

    # Sanity-check for tokenizer.end_of_text_token by sampling a small batch and counting number of string occurrances
    tokenized_dataset_mmaped = np.load(TOKENIZED_FILE_VALIDATION, mmap_mode="r+")
    assert tokenized_dataset_mmaped.dtype == np.uint16
    batch = get_batch(
        tokenized_dataset_mmaped,
        batch_size=32,
        context_length=128,
        device="cpu",
        strategy=SamplingStrategy.SEQ_NON_OVERLAPPING,
    )
    batchwise_token_lists = batch[0].tolist()
    endoftext_str_count = 0
    endoftext_token_count = 0
    for l in batchwise_token_lists:
        decoded_str = tokenizer.decode(l)
        endoftext_str_count += decoded_str.count(tokenizer.end_of_text_str)
        endoftext_token_count += l.count(tokenizer.end_of_text_token)
    assert endoftext_str_count > 0, "Increase batch size and recheck"
    assert (
        endoftext_str_count == endoftext_token_count
    ), "Token count should match string count"

    ### Tinystories:
    # Compression-ratio (bytes / tokens) 4.15 --> We can get a byte-compression of 2x if stored as uint16 (each token < 65k)
    # Throughput-ratio: 1.65 MB/s or 6 GB/hr

    # Load tokenized data. We use Unix's memory mapped mode and create a writeable array which can be converted to torch.tensors
