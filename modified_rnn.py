import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Add gradient checkpointing
from torch.utils.checkpoint import checkpoint

class SimpleRNN(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        self.query_proj = nn.Linear(hidden_dim, key_dim)  # query projection
        self.key_proj = nn.Linear(hidden_dim, key_dim)  # key projection
        self.value_proj = nn.Linear(hidden_dim, value_dim)  # value projection
        self.gate_proj = nn.Linear(hidden_dim, key_dim)  # gate projection
        self.out_proj = nn.Linear(value_dim, output_dim)  # output projection

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: Tensor of shape (T, hidden_dim)

        Returns:
            output: Tensor of shape (T, output_dim)
            final_state: Tensor of shape (key_dim, value_dim)
        """
        T, _ = hidden_state.shape
        K, V = self.key_dim, self.value_dim
        dtype = hidden_state.dtype
        device = hidden_state.device

        # Initialize state once
        state = torch.zeros(K, V, dtype=dtype, device=device)
        output_list = []  # Use list to avoid pre-allocating large tensor
        
        # Process sequence step by step to reduce memory usage
        for i in range(T):
            # Compute projections for current timestep only
            h_i = hidden_state[i:i+1]  # (1, hidden_dim) - keeps batch dimension
            
            query_i = self.query_proj(h_i).squeeze(0)  # (key_dim)
            key_i = torch.sigmoid(self.key_proj(h_i).squeeze(0))  # (key_dim)
            gate_i = torch.sigmoid(self.gate_proj(h_i).squeeze(0))  # (key_dim)
            value_i = self.value_proj(h_i).squeeze(0)  # (value_dim)

            # Compute key-value update using in-place operations where possible
            key_value_i = torch.outer(key_i, value_i)  # More efficient than unsqueeze + multiply
            
            # Update state in-place
            state.mul_(gate_i.unsqueeze(-1))  # In-place multiplication
            state.add_(key_value_i)  # In-place addition
            
            # Compute output for current timestep
            output_i = torch.mv(state.t(), query_i)  # Matrix-vector product, more efficient
            output_list.append(output_i)

        # Stack outputs at the end
        output = torch.stack(output_list, dim=0)  # (T, value_dim)
        return self.out_proj(output), state

class LM(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, num_layers: int,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Embedding layer to convert input_ids to hidden states
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Stack of SimpleRNN layers
        self.layers = nn.ModuleList([SimpleRNN(hidden_dim, key_dim, value_dim, output_dim) for _ in range(num_layers)])

        # Final output projection after reduction
        self.lm_head = nn.Linear(output_dim, vocab_size)  # Output logits for the vocab_size

    def _forward_layer(self, layer, hidden_state):
        """Helper function for gradient checkpointing"""
        output, _ = layer(hidden_state)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (T,) containing token indices

        Returns:
            output: Tensor of shape (T, vocab_size) containing token logits
        """
        hidden_state = self.embedding(input_ids)  # (T, hidden_dim)

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to trade compute for memory during training
                hidden_state = checkpoint(self._forward_layer, layer, hidden_state, use_reentrant=False)
            else:
                hidden_state, _ = layer(hidden_state)  # Pass through each SimpleRNN layer

        output = self.lm_head(hidden_state)  # (T, vocab_size)

        return output


def main():
    vocab_size = 512
    hidden_dim = 128
    key_dim = 32
    value_dim = 64
    output_dim = 128
    num_layers = 2

    model = LM(vocab_size, hidden_dim, key_dim, value_dim, output_dim, num_layers)

    print(model)


if __name__ == "__main__":
    main()
