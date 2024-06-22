import os
import argparse
import torch
import wandb  # For logging losses

from transformer_lm import TransformerModel
from training import cross_entropy_loss, AdamW, lr_cosine_scheduling, gradient_clipping
from training import save_checkpoint, load_checkpoint


def get_device():
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def train(args, name: str):
    wandb.init(project=f"cs336-train-{name}")

    model = TransformerModel(
        args.vocab_size,
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

    # Load data-set from args.data_path and estimate |sample_size|
    sample_size = 99999
    # Initialize data-loader and estimate sample-size

    iters_per_epochs = int(sample_size / float(args.batch_size))
    max_num_iters = float(args.total_train_tokens) / (
        float(args.batch_size) * float(args.context_length)
    )
    total_epochs = max_num_iters / iters_per_epochs
    print(
        f"Training configured for {max_num_iters} iterations with {args.batch_size} batch-size spanning {total_epochs} epochs"
    )

    # Load checkpoint if exists
    start_iter = 0
    if os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path)
        print(f"Checkpoint loaded. Resuming training from iteration {start_iter}.")

    checkpoint_freq = 1000  # iters

    # Compute Epoch
    for niter in range(start_iter, num_train_steps):
        # Get data
        x, y = get_batch(B=B)

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
                niter, args.lr_max, args.lr_min, args.warmup_iters, iters_per_epochs * 5
            ),
        ):
            param_group["lr"] = lr

        current_lr = optimizer.param_groups[0]["lr"]
        niters = lr_scheduler.last_epoch + 1
        print(f"Iter {lr_scheduler.last_epoch + 1}, Learning Rate: {current_lr}")

        # Finally update model params based on gradient
        optimizer.step()

        # Reset gradients / Prevent accumulation for supervised learning:
        # When you call loss.backward() to compute gradients,
        # these gradients accumulate in the .grad attribute of each parameter.
        # This behavior is useful when you want to accumulate gradients over multiple batches
        # (e.g., in reinforcement learning or certain optimization techniques) but we don't want it here.
        optimizer.zero_grad(set_to_none=True)

        if niters % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, model, args.checkpoint_path)


def main():
    """
    Parses CLI args and calls train()
    """
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
    trainer_cli.add_argument("checkpoint_path", help="Path to checkpoint.")
    trainer_cli.add_argument("data_path", help="Path to training data.")
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

    args = parser.parse_args()

    # print(f'Input file: {args.input_file}')
    # if args.verbose:
    #     print('Verbose mode is on')


if __name__ == "__main__":
    main()
