from cs336_basics.transformer.transformer_lm import TransformerModel, softmax
from cs336_basics.transformer.training import initialize_model_from_checkpoint
from cs336_basics.transformer.common import get_device
from cs336_basics.bpe_tokenizer.encoder_decoder import BpePretrainedTokenizer
import torch
import pathlib
import argparse
import os

from typing import Dict, List


# Show how nucleus sampling overcomes "repetitiveness" caused by positive feedback loops in greedy strategies (even beam-search),
# Further, it's much easier for decoder since it doesn't save state during inference and we need to only generate one at a time.
#
# title={The Curious Case of Neural Text Degeneration},
# author={Ari Holtzman and Jan Buys and Li Du and Maxwell Forbes and Yejin Choi},
# year={2020},
# url={https://arxiv.org/abs/1904.09751},
def nucleus_sampling(probabilities: torch.FloatTensor, p: float):
    """
    Perform nucleus (top-p) sampling on a batch of probability distributions.
    Args:
        probabilities (torch.Tensor): A 2D tensor of shape (batch_size, vocab_size) where each row represents a probability distribution.
        p (float): The nucleus sampling threshold. The higher the number, the less truncated the sampling.
    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, 1) consisting of sampled token indices.
    """
    assert 0.0 < p <= 1.0, "Nucleus sampling threshold must be between 0 and 1"

    # Unused, but for readability.
    batch_size, vocab_size = probabilities.shape

    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)

    # Create a mask to filter out tokens where cumulative probability along vocab-dim exceeds p.
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= p
    # Note that the mask must be shifted right because the final element that
    # gets the probability to exceed p is to be included.
    mask[..., 1:] = mask[..., :-1].clone()
    # Ensure that the largest token is always included in sampling
    mask[..., 0] = True

    # Apply the mask to filter probabilities, normalize and sample
    filtered_probs = sorted_probs * mask.float()
    normalized_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(normalized_probs, num_samples=1)

    # Convert sorted indices to original indices
    final_indices = sorted_indices.gather(1, sampled_indices)
    return final_indices


# NOTE: (useful when providing a heavily used service): It's possible to batch prompt requests to the model
# to improve inference compute utlization. The current approach of sequentially generating output is pretty wasteful
# NOTE: We choose not to complicate the decoding and allow the model to genererate text even after end-token is generated.
class Decoder:
    def __init__(
        self,
        tokenizer: BpePretrainedTokenizer,
        max_tokens: int = 1000,
        softmax_temp: float = 1.0,
        nucleus_sampling_threshold: float = 1.0,
    ):
        # Closer to 0 means max. Closer to 1 is standard softmax
        assert 0.0 <= softmax_temp <= 1.0, "Softmax_temp must be between 0 and 1"

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        # For smaller models with more perplexity, it helps to be more "assertive while decoding"
        self.softmax_temp = softmax_temp
        self.nucleus_sampling_threshold = nucleus_sampling_threshold

        assert self.tokenizer.end_of_text_token is not None

    def to_model_input(
        self, prompts: List[str], context_length: int
    ) -> torch.LongTensor:
        # Produce a tensor of shape (batch_size, context_length).
        # The prefix can be padded with self.tokenizer.end_of_text_token
        model_input = (
            torch.ones(size=(len(prompts), context_length), dtype=torch.long)
            * self.tokenizer.end_of_text_token
        )

        for idx, prompt in enumerate(prompts):
            # Encode prompt. If #tokens > context_length, issue a warning.
            token_list = self.tokenizer.encode(prompt)
            if len(token_list) > context_length:
                print(
                    f"WARNING: Truncating first {len(token_list) - context_length} tokens in prompt"
                )
                token_list = token_list[-context_length:]

            # breakpoint()

            # The prompt should be the last part of the context_length.
            model_input[idx, -len(token_list) :] = torch.tensor(token_list[:])

        return model_input

    def decode_prompts(
        self, prompts: List[str], transformer: TransformerModel
    ) -> Dict[int, str]:
        batch_size = len(prompts)
        context_length = transformer.initialization_config.context_length
        model_input = self.to_model_input(prompts, context_length)
        assert (batch_size, context_length) == model_input.shape

        # Initialize
        terminated_prompts = set()
        output_tokens: Dict[int, List[int]] = {
            idx: list() for idx in range(len(prompts))
        }

        # Decode for self.max_tokens steps
        for i in range(self.max_tokens):
            # Run forward pass of transformer to obtain prediction logits
            pred_logits = transformer(model_input)

            # breakpoint()

            # For every batch, only consider the last seq element.
            # pred_logits[:, -1, :] --> (batch_size, vocab_size)
            next_pred_prob = softmax(
                pred_logits[:, -1, :], d_idx=1, temp=self.softmax_temp
            )
            next_tokens = nucleus_sampling(
                next_pred_prob, self.nucleus_sampling_threshold
            )
            print(f"Generated token {i}: {next_tokens}")
            # breakpoint()

            # Process generated tokens
            next_token_list = next_tokens.squeeze(dim=1).tolist()
            for prompt_idx, next_token in enumerate(next_token_list):
                # Skip terminated prompts
                if prompt_idx in terminated_prompts:
                    continue
                output_tokens[prompt_idx].append(next_token)
                # If end-of-text token is reached, mark as terminated
                if next_token == self.tokenizer.end_of_text_token:
                    terminated_prompts.add(prompt_idx)

            # Terminate if all prompts have been decoded
            if len(terminated_prompts) == batch_size:
                print(f"Finished decoding {batch_size} prompts after {i} tokens!")
                break

            # Create a new model_input by shifting less and adding the next tokens.
            # NOTE: If batch_size > 1, this will still include previously terminated batches.
            model_input = torch.cat((model_input[..., 1:], next_tokens), dim=-1)

        # Decode output tokens
        result = {
            prompt_idx: self.tokenizer.decode(decoded_token_list)
            for prompt_idx, decoded_token_list in output_tokens.items()
        }
        assert len(result) == len(prompts)
        return result


### CLI: Run the decoder from a checkpoint


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference from a trained model checkpoint."
    )

    parser.add_argument(
        "--tokenizer_dir", type=str, required=True, help="Path to tokenizer."
    )
    parser.add_argument(
        "--checkpoint_file", type=str, required=True, help="Path to checkpoint."
    )
    parser.add_argument(
        "--max_tokens",
        nargs="?",
        default=256,
        help="Stop decoding after these many tokens even if <|endoftext|> token is not reached",
    )
    parser.add_argument(
        "--softmax_temp",
        nargs="?",
        default=1.0,
        help="Lower values would skew probability mass even more towards higher logits.",
    )
    parser.add_argument(
        "--nucleus_p", nargs="?", default=1.0, help="Nucleus sampling threshold"
    )
    return parser


def main():
    """
    Parses CLI args and runs inference
    """

    args = create_arg_parser().parse_args()

    # Create Tokenizer. We should ensure that this has the same set of special characters as during training.
    dataset_path = pathlib.Path(args.tokenizer_dir)
    vocab_path = str(dataset_path / "vocab.json")
    merge_path = str(dataset_path / "merges.txt")
    tokenizer = BpePretrainedTokenizer.from_files(
        vocab_path, merge_path, special_tokens=["<|endoftext|>"]
    )
    print(f"Tokenizer Ready.")

    # Initialize model and optimizer
    checkpoint_file = pathlib.Path(args.checkpoint_file)
    model = initialize_model_from_checkpoint(str(checkpoint_file)).to(get_device())
    print(f"Model Ready.")

    # Initialize decoder
    decoder = Decoder(
        tokenizer, int(args.max_tokens), float(args.softmax_temp), float(args.nucleus_p)
    )

    # Run decoder on prompt
    prompt = ["Once upon a time there was a little dog Taffy who was very fond of food. Her trainer Lily would give treats every time they went to the park"]
    output = decoder.decode_prompts(prompt, model)
    print(output)


if __name__ == "__main__":
    main()
