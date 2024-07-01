#
## NOTE: This tokenizer from Huggingface is used to train efficiently.
#        We do *not* use this for encoding/decoding:
#           - One reason is that Tokenizer(BPE.from_file(vocab=vocab_path, merges=merge_path, unk_token=UNKNOWN_TOKEN)) isn't working

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import Decoder, BPEDecoder
import pathlib
from typing import List, Optional, Dict
import os
import json


def _post_process_vocab(
    vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None
):
    # Construct vocab dict. from JSON file with format {unicode-coded string: int, }
    with open(vocab_filepath, "r") as file:
        json_vocab_dict = json.load(file)

    with open(merges_filepath, "r") as file:
        for line in file:
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
                if merge_strings[1] == "" and merge_strings[0] not in json_vocab_dict:
                    special_tokens.append(merge_strings[0])

    # If a pretoken is in the vocab, we directly tokenize it and don't take it to the merging process
    if special_tokens:
        for sp in special_tokens:
            json_vocab_dict[sp] = len(json_vocab_dict)

    # Save dictionary to JSON file
    with open(vocab_filepath, "w") as f:
        json.dump(json_vocab_dict, f)


# Define your custom regex pattern
PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
UNKNOWN_TOKEN = "\uFFFD"


def train_tokenizer(
    tokenizer_path: pathlib.Path,
    training_files: List[str],
    vocab_size: int,
    special_phrases: List[str] = [],
):
    # Initialize the tokenizer with the BPE model
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))

    # Set the pre-tokenizer with the custom regex
    tokenizer.pre_tokenizer = Split(
        pattern=Regex(PRETOKEN_PATTERN), behavior="isolated"
    )

    # Define the BPE trainer
    trainer = BpeTrainer(
        special_tokens=[UNKNOWN_TOKEN],
        vocab_size=vocab_size,
    )

    # Train the tokenizer
    tokenizer.train(training_files, trainer)

    # Save the trained tokenizer
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.model.save(str(tokenizer_path))

    # Post training processing: 1) Add special tokens, 2) reconcile with merges
    _post_process_vocab(
        str(tokenizer_path / "vocab.json"),
        str(tokenizer_path / "merges.txt"),
        special_phrases,
    )


# Configure training and post-training phase
dataset_name = "TinyStoriesV2-GPT4"
dataset_path = (
    (pathlib.Path(__file__).resolve()).parent.parent.parent / "data" / dataset_name
)
training_file = dataset_name + "-train.txt"
# Free params
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
train_tokenizer(
    dataset_path,
    [str(dataset_path / training_file)],
    vocab_size,
)


# training_files = [str(file) for file in DATASET_PATH.glob("owt*train.txt")]
# training_files = [str(DATASET_PATH / "owt_train.txt")]
