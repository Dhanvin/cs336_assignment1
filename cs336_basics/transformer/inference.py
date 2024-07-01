from .transformer_lm import TransformerModel, softmax
from ..bpe_tokenizer.encoder_decoder import BpePretrainedTokenizer
import torch
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

            # The prompt should be the last part of the context_length.
            model_input[idx, : -len(token_list)] = torch.tensor(token_list[:])

        return model_input

    def decode_prompts(
        self, prompts: List[str], transformer: TransformerModel
    ) -> Dict[int, str]:
        batch_size = len(prompts)
        context_length = transformer.context_length
        model_input = self.to_model_input(prompts, context_length)
        assert (batch_size, context_length) == model_input.shape

        # Initialize
        terminated_prompts: set = {}
        output_tokens = Dict[int, List[int]] = {idx: [] for idx in range(len(prompts))}

        # Decode for self.max_tokens steps
        for i in range(self.max_tokens):
            # Run forward pass of transformer to obtain prediction logits
            pred_logits = transformer(model_input)

            # For every batch, only consider the last seq element.
            next_pred_prob = softmax(
                pred_logits[:, -1, :], d_idx=2, temp=self.softmax_temp
            )
            next_tokens = nucleus_sampling(
                next_pred_prob, self.nucleus_sampling_threshold
            )

            # Process generated tokens
            next_token_list = next_tokens.squeeze().tolist()
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


# Logic to change
#   1) Add to decoded tokens
#   2) If self.tokenizer.end_of_text_token, then remove that batch from model_input
#   3) If model_input has become empty, exit loop
# NOTE (efficiency for multi-batched): A better way would be to make this a generator of decoded output

# # NOTE: torch.nonzero(mask) is a "hack" to find indices
# # Each row in the output tensor corresponds to the indices where the condition is true.
# terminated = torch.nonzero(next_tokens == self.tokenizer.end_of_text_token)
# # Convert terminated to a tensor of indices to remove
# indices_terminated = torch.tensor(terminated)
# # Use torch.index_select() to select rows not in indices_to_remove
# indices_to_keep = [i for i in range(model_input.shape[0]) if i not in terminated[0,:]]
# model_input_pruned = torch.index_select(model_input, dim=0, index=indices_terminated)
# next_tokens_pruned = torch.index_select(next_tokens, dim=0, index=indices_terminated)


# # Append next tokens to indices_to_keep
# if
# breakpoint()
# if model_input_pruned.numel() == 0:
#     print("Decoding over.")
# if indices_terminated.numel() > 0:
#     print(f"Decoded {indices_terminated.numel()} input prompts!. Remaining: {indices_to_keep.numel()}")
