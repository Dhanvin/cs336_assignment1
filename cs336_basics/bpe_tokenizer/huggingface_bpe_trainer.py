from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import Decoder, BPEDecoder
import pathlib
from typing import List, Optional, Dict
import os

# >> Used to train efficiently and we can use the trainsed
## NOTE: Cannot use this for decoding because Tokenizer(BPE.from_file(vocab=vocab_path, merges=merge_path, unk_token=UNKNOWN_TOKEN)) will not work :(

# Define your custom regex pattern
PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
UNKNOWN_TOKEN = "\uFFFD"

# Initialize the tokenizer with the BPE model
tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))

# Set the pre-tokenizer with the custom regex
tokenizer.pre_tokenizer = Split(pattern=Regex(PRETOKEN_PATTERN), behavior="isolated")

# List of files to train on
DATASET_PATH = (pathlib.Path(__file__).resolve()).parent.parent.parent / "data"
# *GPT4*train.txt
# training_files = [str(file) for file in DATASET_PATH.glob("*.en")]
training_files = [str(file) for file in DATASET_PATH.glob("TinyStories*train.txt")]
# training_files = [str(file) for file in DATASET_PATH.glob("owt*train.txt")]
# training_files = [str(DATASET_PATH / "owt_train.txt")]

# Define the BPE trainer
trainer = BpeTrainer(
    special_tokens=["<|endoftext|>", UNKNOWN_TOKEN],
    vocab_size=32000,  # Adjust as needed
)

# Train the tokenizer
tokenizer.train(training_files, trainer)

# Save the trained tokenizer
model_path = DATASET_PATH / "TinyStories-tokenizer"
vocab_path = str(DATASET_PATH / "vocab.json")
merge_path = str(DATASET_PATH / "merges.txt")
os.makedirs(model_path, exist_ok=True)
tokenizer.model.save(str(model_path))

# Load the tokenizer
# tokenizer = Tokenizer.from_file(str(tokenizer_path))
# loaded_tokenizer = Tokenizer(BPE.from_file(vocab=vocab_path, merges=merge_path, unk_token=UNKNOWN_TOKEN))

# breakpoint()


# # # Create a SafeUnicodeDecoder that wraps the BPEDecoder
# tokenizer.decoder = BPEDecoder()
# # # Set the safe decoder for the tokenizer
# # tokenizer.decoder = safe_bpe_decoder

# # Example text
# text = "Your example text here."
# text = "Your example text here with potential Unicode issues: \uD800"  # Invalid surrogate pair

# # Encode the text
# encoded = tokenizer.encode(text)
# print("Encoded:", encoded.tokens)

# # Decode the tokens back to text
# decoded = tokenizer.decode(encoded.ids)
# print("Decoded:", decoded)

# breakpoint()
