# Implement from scratch using only:
# • torch.nn.Parameter
# • torch.nn.Dropout and torch.nn.functional.dropout
# • torch.nn.Linear and torch.nn.Embedding
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1 • The torch.optim.Optimizer base class.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from dataclasses import dataclass
from .common import get_device


def _approx_gelu(x):
    return (
        0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    )


def gelu_activation(x: torch.FloatTensor):
    """
    Represents a stochastic regularizer on the input
    Loosely, we scale x by how much greater it is than other inputs. This essentially accounts for the idea behind "adaptive dropout"
    Assmumes that neuron inputs tend to follow a normal distribution, especially with Batch Normalization

    Input
    """
    # ? Are we allowed to simply use this library call?
    return torch.nn.functional.gelu(x)
    return _approx_gelu(x)


def softmax(x: torch.FloatTensor, d_idx: int, temp: float = 1.0):
    """
    Applies the softmax
        - For numeric stability, subtract the largest element in that dimension
    Return:
        torch.FloatTensor representing normalized probabilities.
    """
    assert 1e-3 < temp <= 1.0, "Softmax_temp must be between 1e-3 and 1"
    # Apply temperature element-wise
    x = x / temp

    # Compute max and subtract for numeric stability (softmax is invariant to translation).
    max_dim = x.max(dim=d_idx).values
    exp_offset_x = torch.exp(x - max_dim.unsqueeze(d_idx))
    return exp_offset_x / exp_offset_x.sum(d_idx).unsqueeze(d_idx)


def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """
    Inputs
     - Q: queries: (batch_size, ... , Q_seq_length, d_k)
     - K: keys: (batch_size, ... , K_seq_length, d_k)
     - V: values: (batch_size, ... , K_seq_length, d_v)
     - mask: The attention probabilities of the masked positions should be zero, and the relative probabilities
    on the non-masked positions should remain the same.
    """
    # Take dot products of query -> key for each input-token -> target-token pair
    d_k = K.shape[-1]
    attention_weights = (Q @ K.transpose(-1, -2)) / np.sqrt(d_k)

    # Given a mask, set attention weights to -inf where True
    if mask is not None:
        attention_weights.masked_fill_(mask, -np.inf)  # in-place?
    # TODO: Implement effect of pdrop
    # |mask_mul| will be broadcasted since last two dims match.
    # We mask pre-softmax weights to ensure that
    attention_probabilities = softmax(attention_weights, K.dim() - 1)
    if pdrop is not None:
        torch.nn.functional.dropout(attention_probabilities, pdrop, inplace=True)
    return attention_probabilities @ V


# class MyLinear(nn.Module):
#     def __init__(self, d_input: int, d_output: int):
#         super().__init__()
#         # Scale to ensure that we aren't blowing up variance of inputs through the linear transform
#         # Stores weights as transpose to allow for right-matrix-multiply of inputs
#         self.weight = nn.Parameter(torch.randn(d_output, d_input)/ np.sqrt(d_input))

#     def forward(self, x: torch.FloatTensor):
#         return x @ self.weight.t()


class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # gains are named "weight" to match dict key so we can use nn.Module.load_state_dict
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, activations: torch.Tensor):
        """
        Args:
        - activations: torch.FloatTensor
                Input features to run RMSNorm on. Tensor of (*, d_model), where *
                can be an arbitrary number of dimensions with arbitrary values.
        """
        # Unsqueeze reshapes to match dimensionalities so that broadcasting-multiplication happens row-wise
        rms_normalization = (
            activations.pow(2).mean(dim=-1).add(1e-5).rsqrt().unsqueeze(-1)
        )
        return activations * rms_normalization * self.weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # self.w1 = MyLinear(d_model, d_ff)
        # self.w2 = MyLinear(d_ff, d_model)

    def forward(self, x: torch.FloatTensor):
        return self.w2(gelu_activation(self.w1(x)))


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super().__init__()
        # We compress the information based on number of heads (more heads would force more dense subspaces.)
        d_key_val = int(d_model / num_heads)
        # state_dict() conforms to q_heads.{i}.weight for i in ModuleList
        self.q_heads = nn.ModuleList(
            [nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)]
        )
        self.k_heads = nn.ModuleList(
            [nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)]
        )
        self.v_heads = nn.ModuleList(
            [nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)]
        )
        self.output_proj = nn.Linear(d_key_val * num_heads, d_model, bias=False)
        self.attn_pdrop = attn_pdrop

    def forward(self, x: torch.FloatTensor):
        seq_len = x.size(-2)
        last_dim = x.dim() - 1
        future_token_mask = (
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(get_device())
        )
        scaled_attentions = [
            scaled_dot_product_attention(
                self.k_heads[head](x),
                self.q_heads[head](x),
                self.v_heads[head](x),
                future_token_mask,
                self.attn_pdrop,
            )
            for head in range(len(self.q_heads))
        ]
        return self.output_proj(torch.cat(scaled_attentions, last_dim))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
    ):
        super().__init__()
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.ln1 = RmsNorm(d_model)  # layer-normalization
        self.ln2 = RmsNorm(d_model)
        self.residual_dropout = nn.Dropout(residual_pdrop)
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, x: torch.FloatTensor):
        # sub-block1 = x + Dropout(CausalMultiheadSelfAttention(RmsNorm(x)))
        self_attention_output = self.attn(self.ln1(x))
        x = x + self.residual_dropout(self_attention_output)

        # sub-block2 = x + Dropout(PositionwiseFeedForward(RmsNorm(x)))
        feed_forward_ouptut = self.ffn(self.ln2(x))
        output = x + self.residual_dropout(feed_forward_ouptut)
        return output


@dataclass
class TransformerModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    attn_pdrop: float
    residual_pdrop: float
    name: str = "<un-named>"


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerModelConfig):
        super().__init__()
        # Book-keeping. Used for checkpointing.
        self.initialization_config = config

        # Create transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    config.attn_pdrop,
                    config.residual_pdrop,
                )
                for _ in range(config.num_layers)
            ]
        )
        # We are only using this to extract nn.Param for conveneint weight-loading but not in matrix multiply
        # Note that we are using Learned positional and token encoding matrices, which will be updated during training
        self.token_embeddings = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.position_embeddings = nn.Linear(
            config.d_model, config.context_length, bias=False
        )
        self.ln_final = RmsNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )  # Here, should we keep the bias?
        # Tie the weights with self.token_embeddings
        self.lm_head.weight = self.token_embeddings.weight

        self.embedding_dropout = nn.Dropout(config.residual_pdrop)
        self.d_model = config.d_model

    def _get_input_embeddings(self, in_indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            in_indices: Shape is (batch_size, sequence_length)
        Output:
            embeddings: Shape is (batch_size, sequence_length, d_model)
        """
        # Use the "nn.Linear" layers only to extract from weight nn.Params, and not for matrix multiply
        # input_embedding = extracted_embedding from self.token_embeddings.weights and self.position_embeddings.weights
        # Flatten --> torch.gather(expects 1D input Tensor) --> reshape
        seq_len = in_indices.size(1)
        flattened_tokens = in_indices.reshape(-1)
        # breakpoint()
        flattened_embeddings = self.token_embeddings.weight[flattened_tokens]
        assert flattened_embeddings.size(1) == self.d_model, "embedding size mismatch"
        token_embeddings = flattened_embeddings.reshape(
            in_indices.size() + (self.d_model,)
        )
        position_embeddings = self.position_embeddings.weight[:seq_len, :]
        # breakpoint()
        # Will broadcast position_embeddings across the batch dim
        return token_embeddings + position_embeddings

    def forward(self, in_indices: torch.LongTensor):
        """
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

        Returns:
            FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
            next-word distribution for each token.
        """
        input_embedding = self._get_input_embeddings(in_indices)
        x = self.embedding_dropout(input_embedding)
        for transformer_layer in self.layers:
            x = transformer_layer(x)

        # NOTE: We return the unnormalized probabilities before computing softmax
        # for efficient X-entropy loss computation
        return self.lm_head(self.ln_final(x))


############### Accounting for Memory, FLOPs ########################
import yaml


# Accounting of mathmul operations in each Transformer Block:
#   - KVQ Projection: 3 * 2 * L * E * E = 6*L*E^2
#   - scaled_dot_product_attention: QK.t()V: B*(2*L*E/H*L+ 2*L*L*E/H) = 4*L^2*E
#   - Multi-headed causal self-attention (assuming d_v = d_k = d_q = E/H):
#           = H * scaled_dot_product_attention  + B*(2*L*E^2)
#           = B*(4*L^2*E + 2*L*E^2): quadratic in both embedding dim and seq length
#   - 4x Inverted Position-wise Bottleneck (FFN): 16*B*L*E^2
#   --------------------------------------------------------------
#   B* [4*L^2*E + 24*L*E^2]
#
# Accounting of mathmul operations in each Transformer Block:
#   - Position-wise embedding --> Vocab: 2*L*E*V
#   - B * [N*[4*L^2*E + 24*L*E^2]  + 2*L*E*V]
# NOTE: The N*[4*L^2*E] Parameter-free FLOPs makes back-prop very powerful since they can be
#       massively parallelized during training (parameters aren't changing).
#       These can be made ready while the 6*L*E^2 projections and 16*L*E^2 FFN are being computed!
def flops_accounting_fpass(conf: TransformerModelConfig):
    """
    Prints a breakdown of FLOPs accounting per batch element
    """
    # Assumptions:
    #   Assume FFN is a 4x position-wise: Weight matrix is
    #   Key / Query / Value size: E/H

    flops_project_kqv = 6 * conf.context_length * conf.d_model * conf.d_model
    # NOTE: It's awesome to have this parameter-free computation sandwiched between param-rich compute
    flops_self_attention = 4 * conf.context_length * conf.context_length * conf.d_model
    flops_combine_attentions = 2 * conf.context_length * conf.d_model * conf.d_model
    flops_ffn = 16 * conf.context_length * conf.d_model * conf.d_model
    flops_transformer_layer = (
        flops_project_kqv + flops_self_attention + flops_combine_attentions + flops_ffn
    )
    flops_breakdown_transformer = {
        "FLOPs": f"{flops_transformer_layer: .1e} (100%)",
        "breakdown": {
            "project_kqv": f"{flops_project_kqv: .1e} ({flops_project_kqv / flops_transformer_layer *100: .1f}%)",
            "self_attention": f"{flops_self_attention: .1e} ({flops_self_attention / flops_transformer_layer *100: .1f}%)",
            "combine_attentions": f"{flops_combine_attentions: .1e} ({flops_combine_attentions / flops_transformer_layer *100: .1f}%)",
            "ffn": f"{flops_ffn: .1e} ({flops_ffn / flops_transformer_layer *100: .1f}%)",
        },
    }

    flops_all_transformer_layer = conf.num_layers * flops_transformer_layer
    flops_token_encode_or_decode = (
        2 * conf.context_length * conf.d_model * conf.vocab_size
    )
    flops_fpass_total = flops_all_transformer_layer + 2 * flops_token_encode_or_decode

    flops_fpass = {
        "Model": conf.name,
        "FLOPs": f"{flops_fpass_total: .1e} (100%)",
        "breakdown": {
            "encoding": f"{flops_token_encode_or_decode: .1e} ({flops_token_encode_or_decode / flops_fpass_total *100: .1f}%)",
            f"transform_layers ({conf.num_layers} layers)": f"{flops_all_transformer_layer: .1e} ({flops_all_transformer_layer / flops_fpass_total *100: .1f}%)",
            "decoding": f"{flops_token_encode_or_decode: .1e} ({flops_token_encode_or_decode / flops_fpass_total *100: .1f}%)",
        },
        "transformer_layer_details": flops_breakdown_transformer,
    }
    yaml_string = yaml.dump(flops_fpass, default_flow_style=False)
    print(yaml_string)
    return flops_fpass_total


# Memory accounting for params ():
# Transformer Layer: 30M params
#    - FNN: 8* E^2 --> 8 * 1600 * 1600 for GPT-2 XL model params
#    - Multi-HeadSelf-Attention: 4 * E^2
#   -------------------
# 48 Transformer Blocks: 1.5B params for GPT-2 XL model params
#   --------------------
# Output Gen Layer: E*V --> 1600 * 50k for GPT-2 XL: 80M params: 300MB
# Memory foot-print: 1.5 billion * 4 bytes = 6 GB
def params_accounting(conf: TransformerModelConfig):
    params_fnn = 8 * conf.d_model * conf.d_model
    params_self_attention = 4 * conf.d_model * conf.d_model
    params_encode_decode = conf.d_model * conf.vocab_size
    params_total = (
        conf.num_layers * (params_fnn + params_self_attention) + params_encode_decode
    )
    params_model = {
        "Model": conf.name,
        "Params": f"{params_total: .1e} (100%)",
        "breakdown": {
            "FNN": f"{conf.num_layers * params_fnn: .1e} ({conf.num_layers * params_fnn / params_total *100: .1f}%)",
            f"transform_layers ({conf.num_layers} layers)": f"{conf.num_layers * params_self_attention: .1e} ({conf.num_layers * params_self_attention / params_total *100: .1f}%)",
            "decoding": f"{params_encode_decode: .1e} ({params_encode_decode / params_total *100: .1f}%)",
        },
    }
    yaml_string = yaml.dump(params_model, default_flow_style=False)
    print(yaml_string)
    return params_total


def activations_accounting(conf: TransformerModelConfig):
    # - activations (A) per batch B:
    #      -  Transformer block
    #           – RMSNorm(s): 2*L*E
    #           – Multi-head self-attention sublayer:
    #               - QKV projections: L*E
    #               - QKT matrix multiply: L*L
    #               - softmax: L*E
    #               - weighted sum of values: L*E
    #               - output projection: L*E
    #           – Position-wise
    #               - feed-forward: W1 matrix multiply: L*4E
    #               - GELU: L*4E
    #               - W2 matrix multiply: L*E
    #       - final RMSNorm: L*E
    #       - Output embedding: L*V
    #       - Cross-entropy on logits: : L*V
    #   Total A = 15*B*L*E*N + 2*B*N*L*L + 2*B*L*V + B*L*E
    actv_transformer = (
        12 * conf.context_length * conf.d_model
        + 2 * conf.context_length * conf.context_length
    )
    actv_transformer_all = conf.num_layers * actv_transformer
    actv_output_norm = (
        conf.vocab_size * conf.context_length + conf.d_model * conf.context_length
    )
    actv_output_loss = conf.vocab_size * conf.context_length
    actv_total = actv_transformer_all + actv_output_norm + actv_output_loss
    actv_model = {
        "Model": conf.name,
        "Params": f"{actv_total: .1e} (100%)",
        "breakdown": {
            "Transformer-block": f"{actv_transformer_all: .1e} ({actv_transformer_all / actv_total *100: .1f}%)",
            "Output-Predictions": f"{actv_output_norm: .1e} ({actv_output_norm / actv_total *100: .1f}%)",
            "Loss": f"{actv_output_loss: .1e} ({actv_output_loss / actv_total *100: .1f}%)",
        },
    }
    yaml_string = yaml.dump(actv_model, default_flow_style=False)
    print(yaml_string)
    return actv_total


# Running AdamW: Memory consumption:
# - gradients (params and activations) = (A + P)
# - optimizer state = 2 * P
# ---- Total = 2A + 4P = Dominated by 24*B*L*E*N for LLMs
# Notation: batch_size (B) and the model hyperparameters -- vocab_size (V), context_length (L), num_layers (N),d_model (E)
# Memory storage for AdamW: Extra 2 * P, where P is number of params for m, v
def model_training_memory_load(conf: TransformerModelConfig, batch_size: int):
    # - gradients (params and activations) = (A + P)
    # - optimizer state = 2 * P
    n_activations = activations_accounting(conf) * batch_size
    n_params = params_accounting(conf)
    total_floats = 3 * n_activations + 2 * n_params
    backprop_memory_model = {
        "Model": conf.name,
        "Total Mem (GB)": total_floats * 4 / 1.024e9,
        "Params": f"{n_params: .1e} ({n_params / total_floats *100: .1f}%)",
        "Activations": f"{n_activations: .1e} ({n_activations / total_floats *100: .1f}%)",
        "Gradients": f"{n_params + n_activations: .1e} ({(n_params + n_activations) / total_floats *100: .1f}%)",
        "Optimizer State": f"{2 * n_params: .1e} ({2 * n_params / total_floats *100: .1f}%)",
    }
    yaml_string = yaml.dump(backprop_memory_model, default_flow_style=False)
    print(yaml_string)
    return total_floats * 4  # assume single-precision


def model_training_flops(
    conf: TransformerModelConfig, batch_size: int, training_steps: int
):
    fpass_flops = flops_accounting_fpass(conf) * batch_size
    bpass_flops = 2 * fpass_flops
    gradient_update_flops = params_accounting(conf) * 3
    training_flops = training_steps * (
        fpass_flops + bpass_flops + gradient_update_flops
    )

    training_flops_model = {
        "Model": conf.name,
        "Total": f"{training_flops: .1e}",
        "Gradient-Compute-Per-Step": f"{fpass_flops + bpass_flops: .1e}",
        "Param-Update-Per-Step": f"{gradient_update_flops: .1e}",
    }
    yaml_string = yaml.dump(training_flops_model, default_flow_style=False)
    print(yaml_string)
    return training_flops


# CS336 assignment 1: adamwAccounting
def adamwAccounting():
    gpt2_xl_conf = TransformerModelConfig(
        name="GPT-2 XL",
        vocab_size=50257,
        num_layers=48,
        context_length=1024,
        d_model=1600,
    )

    # Similar for forward-pass accounting, memory accounting.
    # flops_accounting(gpt2_xl_conf)
    # model_training_memory_load(gpt2_xl_conf, 4)

    nflops = model_training_flops(gpt2_xl_conf, batch_size=1024, training_steps=400000)
    ns_per_day = 60.0 * 60.0 * 24.0
    flops_s_per_a100_sp = 19.5e12
    model_utilization = 0.5
    flops_per_day = flops_s_per_a100_sp * ns_per_day * model_utilization
    ndays = nflops / flops_per_day
    print(f"{ndays: .1e} days for training {gpt2_xl_conf.name}!")
