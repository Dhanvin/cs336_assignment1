import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
from collections.abc import Callable

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
    assert inputs.dim() == 2, "ERROR: Malformed inputs"
    assert targets.dim() == 1, "ERROR: Malformed targets"
    # Take the vector of raw predictions
    # We take log(sum(exp(x - max_i x_i))), taking row-wise max for numeric stability
    max_elem = torch.max(inputs, dim=1)
    target_logits = inputs[torch.arange(len(targets)), targets] 
    log_prob_seq = max_elem.values - target_logits + torch.log(torch.sum(torch.exp(inputs - max_elem.values.unsqueeze(-1)), dim=1))
    return log_prob_seq.mean()

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
                t = state.get("t", 0) # If state-dict doesn't have t, get 0
                grad = p.grad.data # Set during backward pass called during training
                p.data -=  lr * grad / np.sqrt(t+1) # NOTE: Update parameter data in-place
                state["t"] = t + 1
        return loss # Allows chaining steps.
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
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
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Useful construct provided by base-class to store param specific state
                state = self.state['p']
                
                # Update state based on gradient
                m = state.get('m', torch.zeros(p.data.shape))
                v = state.get('v', torch.zeros(p.data.shape))
                grad = p.grad.data
                state['m'] = betas[0] * m + (1 - betas[0]) * grad
                state['v'] = betas[1] * v + (1 - betas[1]) * (grad ** 2)

                # No idea why this rate is being adjusted
                t = state.get('t', 1) # start from 1
                adjusted_lr = lr * math.sqrt(1 - math.pow(betas[1], t)) / (1. - math.pow(betas[0], t))
                
                # Update params
                p.data -= (adjusted_lr * state['m'] / (torch.sqrt(state['v']) + eps)) # in-place update based on gradient accumulation
                p.data -= (lr * weight_decay * p.data) # in-place weight decay update

                state['t'] = t + 1


        return loss

# Running AdamW: Memory consumption:
# - parameters (P):
#       - FNN: 8* E^2
#       - Multi-Head: 4 * E^2 
#       - Output Gen Layer: E*V
#   Total P = 12E^2 + E*V (indep. of B)
#  
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
#           - final RMSNorm: L*E
#       - Output embedding: L*V
#       - Cross-entropy on logits: : L*V
#   Total A = 12*B*L*E*N + 2*B*L*V + 3*B*L*E
#
# - gradients (params and activations) = (A + P)
# - optimizer state = 2 * P
# ---- Total = 2A + 4P = Dominated by 24*B*L*E*N for LLMs
# Notation: batch_size (B) and the model hyperparameters -- vocab_size (V), context_length (L), num_layers (N),d_model (E)
# Memory storage for AdamW: Extra 2 * P, where P is number of params for m, v