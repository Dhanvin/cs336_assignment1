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

def _approx_gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

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

def softmax(x: torch.FloatTensor, d_idx: int):
    """
    Applies the softma
        - For numeric stability, subtract the largest element in that dimension
    """
    max_dim = x.max(dim=d_idx).values
    # breakpoint()
    exp_offset_x = torch.exp(x - max_dim.unsqueeze(d_idx))
    return exp_offset_x / exp_offset_x.sum(d_idx).unsqueeze(d_idx)

def scaled_dot_product_attention(K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None) -> torch.FloatTensor:
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
        attention_weights.masked_fill_(mask, -np.inf) # in-place?
    # TODO: Implement effect of pdrop    
    # |mask_mul| will be broadcasted since last two dims match.
    # We mask pre-softmax weights to ensure that 
    attention_probabilities = softmax(attention_weights , K.dim() - 1)
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
        rms_normalization = activations.pow(2).mean(dim=-1).add(1e-5).rsqrt().unsqueeze(-1)
        return activations * rms_normalization * self.weight
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 =  nn.Linear(d_model, d_ff, bias=False)
        self.w2 =  nn.Linear(d_ff, d_model, bias=False)
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
        self.q_heads = nn.ModuleList([nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)])
        self.k_heads = nn.ModuleList([nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)])
        self.v_heads = nn.ModuleList([nn.Linear(d_model, d_key_val, bias=False) for _ in range(num_heads)])
        self.output_proj = nn.Linear(d_key_val * num_heads, d_model, bias=False)
        self.attn_pdrop = attn_pdrop

    def forward(self, x: torch.FloatTensor):
        seq_len = x.size(-2)
        last_dim = x.dim()-1
        future_token_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scaled_attentions =[scaled_dot_product_attention(self.k_heads[head](x), self.q_heads[head](x), 
                                                         self.v_heads[head](x), future_token_mask, self.attn_pdrop) 
                                                         for head in range(len(self.q_heads))]
        return self.output_proj(torch.cat(scaled_attentions, last_dim))

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,  attn_pdrop: float, residual_pdrop: float):
        super().__init__()
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.ln1 = RmsNorm(d_model) # layer-normalization
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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int,  attn_pdrop: float, residual_pdrop: float):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        # We are only using this to extract nn.Param for conveneint weight-loading but not in matrix multiply
        self.token_embeddings = nn.Linear(d_model, vocab_size, bias=False) 
        self.position_embeddings = nn.Linear(d_model, context_length, bias=False)
        self.ln_final = RmsNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False) # Here, should we keep the bias?
        self.embedding_dropout = nn.Dropout(residual_pdrop)
        self.d_model = d_model

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
        token_embeddings = flattened_embeddings.reshape(in_indices.size() + (self.d_model, ))
        position_embeddings = self.position_embeddings.weight[:seq_len, :]
        # breakpoint()
        # Will broadcast position_embeddings across the batch dim
        return token_embeddings + position_embeddings

    def forward(self, in_indices: torch.LongTensor):
        """
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
        """
        input_embedding = self._get_input_embeddings(in_indices)
        x = self.embedding_dropout(input_embedding)
        for transformer_layer in self.layers:
            x = transformer_layer(x)

        # NOTE: We return the unnormalized probabilities before computing softmax
        # ? Why are we returning unnormalized? Is it because of X-entropy loss?
        return self.lm_head(self.ln_final(x))

# FLOPS Accounting:
# Assumptions:
#   Input: (B: batch_size, L: seq_len)
#   Embeddings size: E (128 in this case)
#   Assume FFN is a 4x position-wise: Weight matrix is 
#   Num-Heads = H
#   Key / Query / Value size: E/H
#   Vocabulary Size (# Tokens) = V
#   Num Transformer Blocks = N
#
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

# Memory accounting for params ():
# Transformer Layer: 30M params
#    - FNN: 8* E^2 --> 8 * 1600 * 1600 for GPT-2 XL model params
#    - Multi-HeadSelf-Attention: 4 * E^2 
#   -------------------
# 48 Transformer Blocks: 1.5B params for GPT-2 XL model params
#   --------------------
# Output Gen Layer: E*V --> 1600 * 50k for GPT-2 XL: 80M params: 300MB
# Memory foot-print: 1.5 billion * 4 bytes = 6 GB
# 
# Example usage:
# mylib.flops_accounting(mylib.TransformerModelConfiguration(
#     name="GPT-2 XL", 
#     vocab_size=50257,
#     num_layers=48,
#     context_length=1024,
#     embedding_length=1600,
# ))
from dataclasses import dataclass
import yaml

@dataclass
class TransformerModelConfiguration:
    name: str
    vocab_size: int
    num_layers: int
    context_length: int
    embedding_length: int # d_model
    
def flops_accounting(conf: TransformerModelConfiguration):
    """
    Prints a breakdown of FLOPs accounting
   
    """
    # Assumptions:
    #   Assume FFN is a 4x position-wise: Weight matrix is 
    #   Key / Query / Value size: E/H
    
    flops_project_kqv = 6 * conf.context_length * conf.embedding_length * conf.embedding_length
    # NOTE: It's awesome to have this parameter-free computation sandwiched between param-rich compute
    flops_self_attention = 4 * conf.context_length * conf.context_length * conf.embedding_length 
    flops_combine_attentions = 2 * conf.context_length * conf.embedding_length * conf.embedding_length
    flops_ffn = 16 * conf.context_length * conf.embedding_length * conf.embedding_length
    flops_transformer_layer = flops_project_kqv + flops_self_attention + flops_combine_attentions + flops_ffn
    flops_breakdown_transformer = {
        'FLOPs': f"{flops_transformer_layer: .1e} (100%)",
        'breakdown' : {
            'project_kqv': f"{flops_project_kqv: .1e} ({flops_project_kqv / flops_transformer_layer *100: .1f}%)",
            'self_attention': f"{flops_self_attention: .1e} ({flops_self_attention / flops_transformer_layer *100: .1f}%)",
            'combine_attentions': f"{flops_combine_attentions: .1e} ({flops_combine_attentions / flops_transformer_layer *100: .1f}%)",
            'ffn': f"{flops_ffn: .1e} ({flops_ffn / flops_transformer_layer *100: .1f}%)",
        },
    }

    flops_all_transformer_layer = conf.num_layers * flops_transformer_layer
    flops_token_encode_or_decode = 2 * conf.context_length * conf.embedding_length * conf.vocab_size
    flops_fpass_total = flops_all_transformer_layer + 2 * flops_token_encode_or_decode

    flops_fpass = {
        'Model': conf.name,
        'FLOPs': f"{flops_fpass_total: .1e} (100%)",
        'breakdown' : {
            'encoding': f"{flops_token_encode_or_decode: .1e} ({flops_token_encode_or_decode / flops_fpass_total *100: .1f}%)",
            f'transform_layers ({conf.num_layers} layers)': f"{flops_all_transformer_layer: .1e} ({flops_all_transformer_layer / flops_fpass_total *100: .1f}%)",
            'decoding': f"{flops_token_encode_or_decode: .1e} ({flops_token_encode_or_decode / flops_fpass_total *100: .1f}%)",
        },
        'transformer_layer_details': flops_breakdown_transformer,
    }
    yaml_string = yaml.dump(flops_fpass, default_flow_style=False)
    print(yaml_string)


    # Do the same for memory
    # params_fnn = 8* E^2
    # params_project_kqv = 3* E^2
    # params_combine_attentions = E^2
    # params_vocab_prob = 
