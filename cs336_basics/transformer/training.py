import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import numpy.typing as npt
import math
import wandb
from typing import Optional, IO, BinaryIO
from collections.abc import Callable, Iterable
from cs336_basics.transformer.transformer_lm import TransformerModel

import os
from enum import Enum
import random
from .common import SamplingStrategy, get_device, np_array_to_tensor


### Loss Function
def cross_entropy_loss(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.

    NOTE: Perplexity (prob of correct seq) = exp(cross_entropy_loss)
    """
    ## Handle any additional batch dimensions and return the average across the batch.
    assert inputs.dim() - targets.dim() == 1, "ERROR: Malformed inputs"
    inputs = inputs.reshape(-1, inputs.shape[-1])
    targets = targets.reshape(
        -1,
    )
    assert inputs.shape[0] == targets.shape[0], "ERROR: Batch size mismatch"

    # Take the vector of raw predictions
    # We take log(sum(exp(x - max_i x_i))), taking row-wise max for numeric stability
    max_elem = torch.max(inputs, dim=1)
    target_logits = inputs[torch.arange(len(targets)), targets]
    log_prob_seq = (
        max_elem.values
        - target_logits
        + torch.log(torch.sum(torch.exp(inputs - max_elem.values.unsqueeze(-1)), dim=1))
    )
    return log_prob_seq.mean()


### Optimizer
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # Construct default hyperparams to attach to any param-groups without
        # custom metaparams
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)  # If state-dict doesn't have t, get 0
                grad = p.grad.data  # Set during backward pass called during training
                p.data -= (
                    lr * grad / np.sqrt(t + 1)
                )  # NOTE: Update parameter data in-place
                state["t"] = t + 1
        return loss  # Allows chaining steps.


### Optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr: float, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight-decaty rate: {weight_decay}")
        if min(betas) < 0 or max(betas) > 1:
            raise ValueError(f"Invalid beta rate: {betas}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Useful construct provided by base-class to store group specific state
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            # print(f'# params: {len(group["params"])}')
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Useful construct provided by base-class to store param specific state
                state = self.state[p]

                # Update state based on gradient
                m = state.get("m", torch.zeros(p.data.shape).to(get_device()))
                v = state.get("v", torch.zeros(p.data.shape).to(get_device()))
                grad = p.grad.data
                state["m"] = betas[0] * m + (1 - betas[0]) * grad
                state["v"] = betas[1] * v + (1 - betas[1]) * (grad**2)

                # No idea why this rate is being adjusted
                t = state.get("t", 1)  # start from 1
                adjusted_lr = (
                    lr
                    * math.sqrt(1 - math.pow(betas[1], t))
                    / (1.0 - math.pow(betas[0], t))
                )

                # Update params
                p.data -= (
                    adjusted_lr * state["m"] / (torch.sqrt(state["v"]) + eps)
                )  # in-place update based on gradient accumulation
                p.data -= lr * weight_decay * p.data  # in-place weight decay update
                state["t"] = t + 1

        return loss


def lr_cosine_scheduling(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return float(it) * max_learning_rate / float(warmup_iters)
    if it > cosine_cycle_iters:
        return min_learning_rate
    frac_it = float(it - warmup_iters) / float(cosine_cycle_iters - warmup_iters)
    cosine_oscillation = min_learning_rate + 0.5 * (
        max_learning_rate - min_learning_rate
    ) * (1.0 + math.cos(frac_it * math.pi))
    return cosine_oscillation


def combined_gradient_norm(parameters: Iterable[torch.nn.Parameter]):
    # Important Note: L2 norm is computed for all params in the input (entire param-group)
    # Reason: This helps in maintaining the relative magnitudes of the gradients for different parameters,
    # which is important for the training dynamics.
    combined_sum_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        combined_sum_sq += torch.sum(p.grad**2)
    return math.sqrt(combined_sum_sq)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their *combined* gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    # Important Note: L2 norm is computed for all params in the input (entire param-group)
    # Reason: This helps in maintaining the relative magnitudes of the gradients for different parameters,
    # which is important for the training dynamics.
    combined_norm = combined_gradient_norm(parameters)

    if combined_norm > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= max_l2_norm / (combined_norm + 1e-6)
            print(p.grad)


### Data-Loading
def _sample_start_idx(
    min_idx: int,
    max_idx: int,
    batch_size: int,
    context_length: int,
    strategy: SamplingStrategy,
) -> torch.LongTensor:
    """
    Draws |batch_size| samples in [min_idx, max_idx) in continuous chunks of |context_length| based on |strategy|
    Note that max-element in output = max_idx + context_length - 2 (since we are adding two open-intervals)
    """
    output_shape = (batch_size, 1)
    # Use broadcasting of + op to create a 2D tensor
    if strategy == SamplingStrategy.RANDOM:
        return np.random.randint(
            low=min_idx, high=max_idx, size=output_shape
        ) + np.arange(context_length)
    elif strategy == SamplingStrategy.SEQ_NON_OVERLAPPING:
        # Sample
        local_random = random.Random()
        local_random.seed(42)
        idx_list = range(min_idx, max_idx, context_length)
        sampled_with_replacement = local_random.choices(idx_list, k=batch_size)
        return torch.tensor(sampled_with_replacement, dtype=torch.long).view(
            output_shape
        ) + np.arange(context_length)


# NOTE:
# Even though the source data might consist of separate documents (e.g., different web pages, or source code files),
# a common practice is to concatenate all of those into a single sequence of tokens, adding a delimiter between them (such as the <|endoftext|> token).
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    strategy: SamplingStrategy = SamplingStrategy.RANDOM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Check if dataset is memory-mapped
    # if type(dataset) == np.memmap:
    #     print("dataset is memory-mapped.")
    # else:
    #     print("dataset is not memory-mapped.")

    # NOTE: Read-only numpy arrays won't work with torch.from_numpy.
    # This is because torch.tensor will share memory with this array and needs
    # write access.
    assert dataset.flags.writeable

    # Generate input and target idx for the batch
    valid_start_idx = len(dataset) - context_length
    batch_input_idx = _sample_start_idx(
        0, valid_start_idx, batch_size, context_length, strategy
    )
    batch_target_idx = batch_input_idx + 1

    # Convert to tensor. Internally, uint16 will be converted to torch.long
    input_seq = np_array_to_tensor(dataset[batch_input_idx], get_device())
    target_seq = np_array_to_tensor(dataset[batch_target_idx], get_device())

    return (input_seq, target_seq)


### Checkpointing
# str, os.PathLike: string or bytes object representing a file system path
# BinaryIO | IO[bytes]: binary file-like object
def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),  # Same as optimizer.state_dict()
        "niter": iteration,
        "wandb_config": (
            {
                "project": wandb.run.project,
                "run_id": wandb.run.id,
            }
            if wandb.run is not None
            else None
        ),
        "model_init_config": (
            model.initialization_config
            if hasattr(model, "initialization_config")
            else None
        ),
    }
    torch.save(obj, out)


def initialize_model_from_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> TransformerModel:
    obj = torch.load(src, map_location=get_device())
    model = TransformerModel(obj["model_init_config"])
    model.load_state_dict(obj["model_state"])
    return model


def load_model_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])

    # Inference would also load checkpoints but ignore optimizer state.
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer_state"])
    return {
        "niters": obj["niter"],
        "wandb_config": obj["wandb_config"],
        "model_init_config": obj["model_init_config"],
    }
