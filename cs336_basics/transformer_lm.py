# Implement from scratch using only:
# • torch.nn.Parameter
# • torch.nn.Dropout and torch.nn.functional.dropout
# • torch.nn.Linear and torch.nn.Embedding
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1 • The torch.optim.Optimizer base class.

# Input: (batch_size, seq_length) Tensor, where each data point in the batch is a single word in the word-vocabulary, represented as a sequence of tokens
# Output: (batch_size, seq_length, vocab_size) Tensor, next token distribution for each input token sequence

# Token embeddings: V:  vocab_size x d_model (for embedding_vector)
# Position embeddings: U:  context_length x d_model
# NOTE: These embeddings have same dimensionality so they can be added.

# Step 1: Token embeddings: (batch_size, seq_length) --> (batch_size, seq_length, d_model for embedding vector)
#       + Absolute Learned position embeddings: (batch_size, seq_length) --> (batch_size, seq_length, d_model for position vector)
#       --> Apply embedding-dropout
# N x Step 2: Trandformer block: (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
#           - Layer normalization (RMSNorm, which is simpler) is applied to the input of each sub-layer (leads to more stable training than than to the output)
#                - RMSNorm provides invarience to input re-scaling and negative correlation to Weight matrix scaling, which acts as an adaptive rate, improving model convergence (paper)
#           - Sublayers:
#                   - Multi-Head self-attention
#                       - Learnable params: Linear projections from d_model --> d_k (for keys and queries) as well as d_model --> d_v (values).
#                           -- d_k and d_v are smaller in dimension than the embedding vector (64 in case of transformer paper)
#                       - Attention (queries)
#                       - input-vector --> Softmax-normalized(input): subtract out maximal element for numeric stability
#                   - Position-wise feed-forward network for GELU non-linear projection of each position independently of other positions --> shared weights across positions with omitted biases.
#                           -- different from dense fully-connected linear layer, whereas in a dense layer processes the entire input vector, here, each token is processed independently (shared weights across positions)
#           - Dropout is applied to the output of each Transformer block sub-layer, before it is added to the sub-layer
#
# Step 3: Layer normalization + Linear layer + Embedding (batch_size, seq_length, d_model) --> (batch_size, seq_length, vocab_size)

# ? What is encoder-decoder attention v/s decoder only attention v/s encoder-only attention (multi-head self-attention)
#       -- In decoder, prevent positions from attending to subsequent positions using masking
#       -- Multi-head: parallel layers of scaled dot product attention (SDP) all concatenated and linearly transformed, to promote attending to different properties
#       -- The whole idea behind encoder-decoder models is that all info in a seq. is compressed into an encoding vector which is then decoded. This is very limiting.
# ? What is the function of residual connections
# ? Why does the FFN expand to 4x dim and then compress as opposed to compress to lower-dim and then expand?
#

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
    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super().__init__()
        # We compress the information based on number of heads (more heads would force more dense subspaces.)
        d_key_val = int(d_model / num_heads)
        # Attempt to conform to q_heads.{N}.weight
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
    

# class MyLinear(nn.Module):
#     def __init__(self, d_input: int, d_output: int):
#         super().__init__()
#         # Scale to ensure that we aren't blowing up variance of inputs through the linear transform
#         # Stores weights as transpose to allow for right-matrix-multiply of inputs
#         self.weight = nn.Parameter(torch.randn(d_output, d_input)/ np.sqrt(d_input))
    
#     def forward(self, x: torch.FloatTensor):
#         return x @ self.weight.t()

        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 =  nn.Linear(d_model, d_ff, bias=False)
        self.w2 =  nn.Linear(d_ff, d_model, bias=False)
        # self.w1 = MyLinear(d_model, d_ff)
        # self.w2 = MyLinear(d_ff, d_model)
    
    def forward(self, x: torch.FloatTensor):
        return self.w2(gelu_activation(self.w1(x)))
