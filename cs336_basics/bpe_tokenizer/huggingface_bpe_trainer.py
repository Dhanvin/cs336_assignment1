from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import Decoder, BPEDecoder
import pathlib
from typing import List, Optional, Dict
import os

# >> Used to train efficiently and we can use the trainsed
## NOTE: Cannot use this for decoding because 
# Tokenizer(BPE.from_file(vocab=vocab_path, merges=merge_path, unk_token=UNKNOWN_TOKEN)) will not work :(


# Define your custom regex pattern
PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
UNKNOWN_TOKEN = "\uFFFD"

def train_tokenizer(tokenizer_path: pathlib.Path, training_files: List[str], vocab_size: int):
    # Initialize the tokenizer with the BPE model
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))

    # Set the pre-tokenizer with the custom regex
    tokenizer.pre_tokenizer = Split(pattern=Regex(PRETOKEN_PATTERN), behavior="isolated")

    # Define the BPE trainer
    trainer = BpeTrainer(
        special_tokens=["<|endoftext|>", UNKNOWN_TOKEN],
        vocab_size=vocab_size,  
    )

    # Train the tokenizer
    tokenizer.train(training_files, trainer)

    # Save the trained tokenizer
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.model.save(str(tokenizer_path))

# List of files to train on
DATASET_PATH = (pathlib.Path(__file__).resolve()).parent.parent.parent / "data"
training_files = [str(file) for file in DATASET_PATH.glob("TinyStories*train.txt")]
model_path = DATASET_PATH / "TinyStories-tokenizer"
# Adjust as needed
train_tokenizer(model_path, training_files, vocab_size=32000)


# training_files = [str(file) for file in DATASET_PATH.glob("owt*train.txt")]
# training_files = [str(DATASET_PATH / "owt_train.txt")]

