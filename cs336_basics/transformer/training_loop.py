import os
import argparse
import torch
import numpy as np
import numpy.typing as npt
import wandb  # For logging losses
import pathlib
import json
from typing import List, Tuple

from cs336_basics.transformer.transformer_lm import TransformerModel
from cs336_basics.transformer.training import (
    cross_entropy_loss,
    AdamW,
    lr_cosine_scheduling,
    gradient_clipping,
    combined_gradient_norm,
)
from cs336_basics.transformer.common import SamplingStrategy, get_device
from cs336_basics.transformer.training import (
    save_model_checkpoint,
    load_model_checkpoint,
)
from cs336_basics.transformer.training import get_batch, SamplingStrategy


# TODO(slinky):
#   - How should we initilize weights for training, especially for the Position embedding layer and token encoding layer --> randomly.. these are going to be learned too?
#   - model.vocab_size should be derived from tokenizer? What is the relation between model vocabulary size and tokenizer vocabulary size?
#       - Currently, I ensure that I post-process the vocab with special tokens and merge-list  after traning and here, I just load


def get_tokenizer_vocab_size(args):
    # TODO: Get |vocab_size| from tokenizer.
    dataset_path = pathlib.Path(args.dataset_dir)
    vocab_filepath = str(dataset_path / "vocab.json")
    # Construct vocab dict. from JSON file with format {unicode-coded string: int, }
    with open(vocab_filepath, "r") as file:
        json_vocab_dict = json.load(file)
    vocab_size = len(json_vocab_dict)
    return vocab_size


def initialize_model(args) -> TransformerModel:
    model = TransformerModel(
        get_tokenizer_vocab_size(args),
        int(args.context_length),
        int(args.num_layers),
        int(args.d_model),
        int(args.num_heads),
        int(args.d_model) * 4,
        float(args.attn_pdrop),
        float(args.residual_pdrop),
    ).to(get_device())
    return model


def initialize_optimizer(args, model: TransformerModel) -> AdamW:
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(float(args.beta_1), float(args.beta_2)),
        eps=1e-8,
    )
    return optimizer


def compute_train_validation_losses(
    args,
    model: TransformerModel,
    training_dataset: npt.NDArray,
    validation_dataset: npt.NDArray,
):
    # Disable aspects of the model that are train-only (e.g. dropout)
    model.eval()
    print(f"Computing Train and Validaion losses on {args.eval_batch_size} batches...")

    # Sample validation and training data with non-overlapping batches
    val_input_tensor, val_target_tensor = get_batch(
        validation_dataset,
        int(args.eval_batch_size),
        int(args.context_length),
        device=get_device(),
        strategy=SamplingStrategy.SEQ_NON_OVERLAPPING,
    )
    train_input_tensor, train_target_tensor = get_batch(
        training_dataset,
        int(args.eval_batch_size),
        int(args.context_length),
        device=get_device(),
        strategy=SamplingStrategy.SEQ_NON_OVERLAPPING,
    )
    with torch.no_grad():
        val_pred_logits = model(val_input_tensor)
        val_loss = cross_entropy_loss(val_pred_logits, val_target_tensor)

        train_pred_logits = model(train_input_tensor)
        train_loss = cross_entropy_loss(train_pred_logits, train_target_tensor)

    # Reset training-mode for model
    model.train()
    return (train_loss, val_loss)


def train(args):
    checkpoint_dir = pathlib.Path(args.checkpoint_path).resolve()

    # Initialize model and optimizer
    model = initialize_model(args)
    optimizer = initialize_optimizer(args, model)

    # Load from checkpoint if it exists.
    start_iter = 1
    checkpoint_dir = pathlib.Path(args.checkpoint_path)
    checkpoint_file = str(checkpoint_dir / (args.name + "_checkpoint.pt"))
    if os.path.exists(checkpoint_file):
        load_state = load_model_checkpoint(str(checkpoint_file), model, optimizer)
        start_iter = load_state['niters']
        # Initialize wandb with the resume option
        if "wandb_config" in load_state and load_state["wandb_config"] is not None:
            wandb.init(project=load_state["wandb_config"]['project'], 
                    id=load_state["wandb_config"]['run_id'], 
                    resume='allow')
        print(f"Checkpoint loaded. Resuming training from iteration {start_iter}.")
    else:
        wandb.init(project=f"cs336-assignment1",
                   id=f"{args.name}")

    # Load tokenized training data. We use Unix's memory mapped mode and create a writeable array which can be converted to torch.tensors
    dataset_path = pathlib.Path(args.dataset_dir)
    dir_name = dataset_path.stem
    training_data_path = dataset_path / (str(dir_name) + "-train-tokens.npy")
    validation_data_path = dataset_path / (str(dir_name) + "-valid-tokens.npy")
    training_dataset_mmaped = np.load(str(training_data_path), mmap_mode="r+")
    assert training_dataset_mmaped.dtype == np.uint16
    validation_dataset_mmaped = np.load(str(validation_data_path), mmap_mode="r+")
    assert validation_dataset_mmaped.dtype == np.uint16

    # Compute epochs and iteration info. Set cosine LR scheduling niters
    ntokens_per_epoch = len(training_dataset_mmaped)
    ntokens_per_iter = float(args.training_batch_size) * float(args.context_length)
    iters_per_epochs = int(ntokens_per_epoch / ntokens_per_iter)
    cosine_cycle_iters = iters_per_epochs * args.lr_cosine_nepochs
    max_num_iters = int(float(args.total_train_tokens) / ntokens_per_iter)
    training_epochs = max_num_iters / iters_per_epochs
    print(
        f"Training configured for {max_num_iters} iterations with {args.training_batch_size} batch-size spanning {training_epochs} epochs"
    )

    # Start training, checkpointing along the way
    checkpoint_freq = 100  # iters
    validation_loss_freq = 10  # iters
    for niter in range(start_iter, max_num_iters):
        # Get training batch
        input_tensor, target_tensor = get_batch(
            training_dataset_mmaped,
            int(args.training_batch_size),
            int(args.context_length),
            device=get_device(),
            strategy=SamplingStrategy.RANDOM,
        )

        # Update learning rate for the optimizer.
        lr_now = lr_cosine_scheduling(
            niter,
            float(args.lr_max),
            float(args.lr_min),
            int(args.lr_warmup_iters),
            cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # Forward (compute loss)
        pred_logits = model(input_tensor)
        train_loss = cross_entropy_loss(pred_logits, target_tensor)

        # Backward (compute gradients)
        train_loss.backward()
        gradient_clipping(model.parameters(), float(args.max_gradient_norm))
        curr_gradient_norm = combined_gradient_norm(model.parameters())

        # Finally update model params based on gradient
        optimizer.step()

        # Reset gradients / Prevent accumulation for supervised learning:
        # When you call loss.backward() to compute gradients,
        # these gradients accumulate in the .grad attribute of each parameter.
        # This behavior is useful when you want to accumulate gradients over multiple batches
        # (e.g., in reinforcement learning or certain optimization techniques) but we don't want it here.
        optimizer.zero_grad(set_to_none=True)

        # Compute train and validation losses and record them to Weights & Biases
        if niter % validation_loss_freq == 0:
            train_loss, validation_loss = compute_train_validation_losses(
                args, model, training_dataset_mmaped, validation_dataset_mmaped
            )
            log_dict = {
                    "iteration": niter,
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "grad_norm": curr_gradient_norm,
                    "learning_rate": lr_now,
                }
            wandb.log(log_dict)
            print(log_dict)

        # Checkpoint
        if niter % checkpoint_freq == 0:
            parent_dir = os.path.dirname(checkpoint_file)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
                print(f"Created directories up to {parent_dir}")
            print(f"Saving checkpoint at {checkpoint_file}")
            save_model_checkpoint(model, optimizer, niter, str(checkpoint_file))

    wandb.finish()


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The training loop for Transformer Model using AdamW optimizer."
    )

    # Create argument groups for better --help.
    scheduler_cli = parser.add_argument_group(
        "LR Scheduler CLI params", "Cosine Annealing Learning-Rate schedule params."
    )
    optimizer_cli = parser.add_argument_group(
        "Optimier CLI params", "AdamW optimizer hyper-parameters."
    )
    model_cli = parser.add_argument_group(
        "Model CLI params", "Transformer Model hyper-parameters."
    )
    trainer_cli = parser.add_argument_group(
        "Info needed for traning",
        "File paths for dataset and checkpointing, Batch-Size, and Token-processing-upper-limit.",
    )

    # Required positional arguments with default values specified
    scheduler_cli.add_argument(
        "--lr_min",
        nargs="?",
        default=1e-5,
        help="Min learning rate for cosine-scheduler. 10^-4 is a good default.",
    )
    scheduler_cli.add_argument(
        "--lr_max",
        nargs="?",
        default=1e-4,
        help="Max learning rate for cosine-scheduler. 10^-4 is a good default.",
    )
    scheduler_cli.add_argument(
        "--lr_warmup_iters",
        nargs="?",
        default=500,
        help="Warmup iterations for cosine-scheduler. 10 percent of iters is a good default",
    )
    scheduler_cli.add_argument(
        "--lr_cosine_nepochs",
        nargs="?",
        default=5,
        help="Number of epochs for cosine pattern",
    )

    optimizer_cli.add_argument(
        "--beta_1",
        nargs="?",
        default=0.9,
        help="Beta for first-order gradient momentum (g) aggregation",
    )
    optimizer_cli.add_argument(
        "--beta_2",
        nargs="?",
        default=0.999,
        help="Beta for second-order gradient momentum (g^2) aggregation",
    )
    optimizer_cli.add_argument(
        "--max_gradient_norm", nargs="?", default=1.0, help="Gradient clipping norm"
    )

    # Model hyper-params
    model_cli.add_argument(
        "--context_length",
        nargs="?",
        default=256,
        help="Context length == sequence length",
    )
    model_cli.add_argument(
        "--d_model", nargs="?", default=512, help="Embedding dimensionality"
    )
    model_cli.add_argument(
        "--num_layers", nargs="?", default=4, help="Number of transformer layers."
    )
    model_cli.add_argument(
        "--num_heads",
        nargs="?",
        default=16,
        help="Number of heads for multi-head transformer",
    )
    model_cli.add_argument(
        "--attn_pdrop",
        nargs="?",
        default=0.2,
        help="Dropout rate for attention-probabilities",
    )
    model_cli.add_argument(
        "--residual_pdrop", nargs="?", default=0.2, help="Dropout rate for residuals"
    )

    # Info for training
    trainer_cli.add_argument("--name", help="Name given to training run.")
    trainer_cli.add_argument("--checkpoint_path", help="Path to checkpoint.")
    trainer_cli.add_argument(
        "--dataset_dir",
        help="Directory to dataset. Assumes that we have the following files inside this dir: <dir>-train.txt, merges.txt, vocab.json, <dir>-valid.txt, <dir>-train-tokens.npy, <dir>-valid-tokens.npy.",
    )
    trainer_cli.add_argument("--training_batch_size", nargs="?", default=4, help="")
    trainer_cli.add_argument(
        "--eval_batch_size",
        nargs="?",
        default=10,
        help="Number of batches we want to sample for validation",
    )
    trainer_cli.add_argument(
        "--total_train_tokens",
        nargs="?",
        default=10000,
        help="Upper limit on training: batch size * total_step_count * context length",
    )

    # Optional arguments
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    return parser


def main():
    """
    Parses CLI args and calls train()
    """
    args = create_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
