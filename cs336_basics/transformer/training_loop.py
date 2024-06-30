import os
import argparse
import torch
import numpy as np
import wandb  # For logging losses
import pathlib
import json

from cs336_basics.transformer.transformer_lm import TransformerModel
from cs336_basics.transformer.training import (
    cross_entropy_loss,
    AdamW,
    lr_cosine_scheduling,
    gradient_clipping,
)
from cs336_basics.transformer.training import save_checkpoint, load_checkpoint
from cs336_basics.transformer.training import get_batch

# TODO(slinky):
#   - How should we initilize weights for training, especially for the Position embedding layer and token encoding layer
#   - model.vocab_size should be derived from tokenizer? What is the relation between model vocabulary size and tokenizer vocabulary size?
#       - Currently, I ensure that I post-process the vocab with special tokens and merge-list  after traning and here, I just load

# TODO(dhanvin):
#   - Move the vocab <> merges reconciliation logic that's currently in the encoder directly into the tokenization trainer (huggingface).
#       - This will ensure that the saved vocab
#


def get_device():
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def train(args, experiment_name: str):
    wandb.init(project=f"cs336-train-{experiment_name}")

    # TODO: Get |vocab_size| from tokenizer.
    dataset_path = pathlib.Path(args.dataset_dir)
    vocab_filepath = str(dataset_path / "vocab.json")
    # Construct vocab dict. from JSON file with format {unicode-coded string: int, }
    with open(vocab_filepath, "r") as file:
        json_vocab_dict = json.load(file)
    vocab_size = len(json_vocab_dict)

    model = TransformerModel(
        vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_model * 4,
        args.attn_pdrop,
        args.residual_pdrop,
    ).to(get_device())

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
    )

    # Load tokenized data. We use Unix's memory mapped mode and create a writeable array which can be converted to torch.tensors
    training_dataset_mmaped = np.load(args.training_dataset, mmap_mode="r+")
    assert training_dataset_mmaped.dtype == np.uint16

    ntokens_per_epoch = len(training_dataset_mmaped)
    ntokens_per_iter = float(args.batch_size) * float(args.context_length)
    iters_per_epochs = int(ntokens_per_epoch / ntokens_per_iter)

    max_num_iters = float(args.total_train_tokens) / ntokens_per_iter
    total_epochs = max_num_iters / iters_per_epochs
    print(
        f"Training configured for {max_num_iters} iterations with {args.batch_size} batch-size spanning {total_epochs} epochs"
    )

    # Set to 5 epochs (don't know why)
    cosine_cycle_iters = iters_per_epochs * 5

    # Load checkpoint if exists
    start_iter = 0
    if os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path)
        print(f"Checkpoint loaded. Resuming training from iteration {start_iter}.")

    checkpoint_freq = 1000  # iters

    # Start training
    for niter in range(start_iter, max_num_iters):
        # Get data
        x, y = get_batch(
            training_dataset_mmaped, args.batch_size, args.context_length, device="cpu"
        )

        # Forward (compute loss)
        pred_y = model(x)
        loss = cross_entropy_loss(pred_y, y)
        wandb.log({"loss": loss.item()})

        # Backward (compute gradients)
        loss.backward()

        # Update parameters
        gradient_clipping(model.parameters(), args.max_gradient_norm)

        # Update learning rate and check that it reflects in
        for param_group, lr in zip(
            optimizer.param_groups,
            lr_cosine_scheduling(
                niter, args.lr_max, args.lr_min, args.warmup_iters, cosine_cycle_iters
            ),
        ):
            param_group["lr"] = lr

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Iter {niter}, Learning Rate: {current_lr}")

        # Finally update model params based on gradient
        optimizer.step()

        # Reset gradients / Prevent accumulation for supervised learning:
        # When you call loss.backward() to compute gradients,
        # these gradients accumulate in the .grad attribute of each parameter.
        # This behavior is useful when you want to accumulate gradients over multiple batches
        # (e.g., in reinforcement learning or certain optimization techniques) but we don't want it here.
        optimizer.zero_grad(set_to_none=True)

        if niter % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, niter, args.checkpoint_path)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The training loop for Transformer Model using AdamW regularizer."
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
        "lr_min", nargs="?", default=1e-3, help="Min learning rate for cosine-scheduler"
    )
    scheduler_cli.add_argument(
        "lr_max", nargs="?", default=1e-2, help="Max learning rate for cosine-scheduler"
    )
    scheduler_cli.add_argument(
        "warmup_iters",
        nargs="?",
        default=50,
        help="Warmup iterations for cosine-scheduler",
    )

    optimizer_cli.add_argument(
        "beta_1",
        nargs="?",
        default=0.9,
        help="Beta for first-order gradient momentum (g) aggregation",
    )
    optimizer_cli.add_argument(
        "beta_2",
        nargs="?",
        default=0.999,
        help="Beta for second-order gradient momentum (g^2) aggregation",
    )
    optimizer_cli.add_argument(
        "max_gradient_norm", nargs="?", default=1.0, help="Gradient clipping norm"
    )

    # Model hyper-params
    model_cli.add_argument(
        "vocab_size", nargs="?", default=10000, help="Vocabulary size (num tokens)"
    )
    model_cli.add_argument(
        "context_length",
        nargs="?",
        default=1024,
        help="Context length == sequence length",
    )
    model_cli.add_argument(
        "d_model", nargs="?", default=1024, help="Embedding dimensionality"
    )
    model_cli.add_argument(
        "num_layers", nargs="?", default=24, help="Number of transformer layers."
    )
    model_cli.add_argument(
        "num_heads",
        nargs="?",
        default=8,
        help="Number of heads for multi-head transformer",
    )
    model_cli.add_argument(
        "attn_pdrop",
        nargs="?",
        default=0.0,
        help="Dropout rate for attention-probabilities",
    )
    model_cli.add_argument(
        "residual_pdrop", nargs="?", default=0.0, help="Dropout rate for residuals"
    )

    # Info for training
    trainer_cli.add_argument("name", help="Name given to training run.")
    trainer_cli.add_argument("checkpoint_path", help="Path to checkpoint.")
    trainer_cli.add_argument(
        "dataset_dir",
        help="Directory to dataset. Assumes that we have the following files inside this dir: <dir>-train.txt, merges.txt, vocab.json, <dir>-valid.txt, <dir>-train-tokens.npy, <dir>-valid-tokens.npy.",
    )
    trainer_cli.add_argument("batch_size", nargs="?", default=4, help="")
    trainer_cli.add_argument(
        "total_train_tokens",
        nargs="?",
        default=10000,
        help="Upper limit on training: batch size * total_step_count * context length",
    )

    # Optional arguments
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )


def main():
    """
    Parses CLI args and calls train()
    """
    args = create_arg_parser().parse_args()
    train(args, "test")


if __name__ == "__main__":
    main()
