# Implements a BPE encoder and decoder

# Input:
#   merges.txt file, with an frequency-ordered set of merges during training
#   vocab.json file, a mapping from tokens to
from typing import List, Dict, Tuple, Iterable, Iterator, Generator
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
        for i in all_indices:
            mutating_text_list[i] = " "
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
        # Sort keywords by length in descending order. This will prioritize merges peroperly in |pattern_match_with_special_keywords()|
        self.sorted_special_tokens = (
            sorted(special_tokens, key=len, reverse=True)
            if special_tokens is not None
            else None
        )

        # If a pretoken is in the vocab, we directly tokenize it and don't take it to the merging process
        if special_tokens:
            for sp in special_tokens:
                sp_b = sp.encode("utf-8")
                self.vocab[len(self.vocab)] = sp_b

        # breakpoint()
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

        self.pretoken_regex = re.compile(PRETOKEN_PATTERN)
        # Prioritized list of compiled regular expressions
        if self.sorted_special_tokens:
            self.regex_list = [
                re.compile(rf"{re.escape(word)}") for word in self.sorted_special_tokens
            ]
            self.regex_list += [re.compile(PRETOKEN_PATTERN)]  # Lowest priority
        else:
            self.regex_list = [re.compile(PRETOKEN_PATTERN)]

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
                    # NOTE: If the line is a sequence of white-spaces, we lose information so we cannot use
                    # the merge-pair ordering during encoding. Instead, we collect these special strings and
                    # pass it to Utf8PreTokenTokenPairs during initial tokenization of the pretoken prior to
                    # kicking off the encoding algorithm
                    if merge_strings[1] == "":
                        special_tokens.append(merge_strings[0])
                    else:
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

        print(f"Registering special tokens to vocab for encoding: {special_tokens}")
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
        # print(f"Pretokens generated: {pretokens}")

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
                print("Encoding over: No token pairs to process")
                break
            selected_token_pair = min(
                token_pair_map.token_pair_corpus_info.keys(),
                key=_safe_token_pair_ranker,
            )
            if selected_token_pair not in self.token_pair_merge_ranking:
                print("Encoding over: No more merges possible")
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
        while True:
            if token_buffer:
                yield token_buffer.pop(0)  # Remove and return in order of insertion
            else:
                try:
                    line = next(
                        iterable
                    )  # If iterable is a file in 'r' mode, this returns a line
                except StopIteration:
                    print("Iterator has ended.")
                    break
                token_buffer += self.encode(line)
                yield token_buffer.pop(0)

    # Decode a sequence of token IDs into text.
    def decode(self, ids: List[int]) -> str:
        utf8_str = b"".join([self.vocab[token] for token in ids])
        return utf8_str.decode("utf-8", errors="replace")


# python -m cs336_basics.bpe_tokenizer.encoder_decoder
if __name__ == "__main__":
    # TOKENIZER_PATH = (
    #     (pathlib.Path(__file__).resolve()).parent.parent.parent
    #     / "data"
    #     / "TinyStories-tokenizer"
    # )
    # vocab_path = str(TOKENIZER_PATH / "vocab.json")
    # merge_path = str(TOKENIZER_PATH / "merges.txt")

    # tokenizer = BpePretrainedTokenizer.from_files(
    #     vocab_path, merge_path, special_tokens=["<|endoftext|>"]
    # )
    # text = "This is awesome     yablasdf32"
    # encoded_tokens = tokenizer.encode(text)
    # encoded_byte_seq = [tokenizer.vocab[token] for token in encoded_tokens]
    # print(f"Encoded: '{text}' --> {encoded_tokens} == {encoded_byte_seq}")

    TOKENIZER_PATH = (
        (pathlib.Path(__file__).resolve()).parent.parent.parent / "tests" / "fixtures"
    )
    vocab_path = str(TOKENIZER_PATH / "gpt2_vocab.json")
    merge_path = str(TOKENIZER_PATH / "gpt2_merges.txt")

    tokenizer = BpePretrainedTokenizer.from_files(
        vocab_path, merge_path, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    # breakpoint()
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    breakpoint()
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
