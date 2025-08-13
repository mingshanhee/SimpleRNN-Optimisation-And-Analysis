import torch
import torch.nn as nn
from typing import Optional, Tuple

# Add gradient checkpointing
from torch.utils.checkpoint import checkpoint

class SimpleRNN(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            key_dim: int, 
            value_dim: int, 
            output_dim: int,
            fused_projection: bool = True
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.fused_projection = fused_projection

        if fused_projection:
            # Single projection layer for all components
            self.proj = nn.Linear(hidden_dim, 3 * key_dim + value_dim)
        else:
            # Separate projection layers for each component
            self.query_proj = nn.Linear(hidden_dim, key_dim)  # query projection
            self.key_proj = nn.Linear(hidden_dim, key_dim)  # key projection
            self.value_proj = nn.Linear(hidden_dim, value_dim)  # value projection
            self.gate_proj = nn.Linear(hidden_dim, key_dim)  # gate projection

        self.out_proj = nn.Linear(value_dim, output_dim)  # output projection

    def forward(
            self, 
            hidden_state: torch.Tensor,
            cell_state: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: Tensor of shape (B, T, hidden_dim)

        Returns:
            output: Tensor of shape (B, T, output_dim)
            final_state: Tensor of shape (B, key_dim, value_dim)
        """
        B, T, _ = hidden_state.shape
        K, V = self.key_dim, self.value_dim
        dtype = hidden_state.dtype
        device = hidden_state.device

        # Initialize cell state if not provided
        if cell_state is None:
            cell_state = torch.zeros(B, K, V, dtype=dtype, device=device)  # (B, K, V)
        
        # Use list to avoid pre-allocating large tensor
        outputs = torch.empty(B, T, self.value_dim, dtype=dtype, device=device)
        
        # Process sequence step by step to reduce memory usage
        for i in range(T):
            # Compute projections for current timestep
            h_i = hidden_state[:, i, :]  # (B, hidden_dim)
            
            if self.fused_projection:
                # Single projection for all components
                qkgv = self.proj(h_i)
                query_i, key_i, gate_i, value_i = torch.split(qkgv, [K, K, K, V], dim=-1)
                key_i  = torch.sigmoid(key_i)
                gate_i = torch.sigmoid(gate_i)
            else:
                query_i = self.query_proj(h_i)  # (B, key_dim)
                key_i = torch.sigmoid(self.key_proj(h_i))  # (B, key_dim)
                gate_i = torch.sigmoid(self.gate_proj(h_i))  # (B, key_dim)
                value_i = self.value_proj(h_i)  # (B, value_dim)

            # Compute key-value update - batch matrix multiplication
            key_value_i = torch.bmm(key_i.unsqueeze(-1), value_i.unsqueeze(1))  # (B, key_dim, value_dim)
            
            # Update cell_state (non-in-place to avoid gradient issues)
            cell_state = cell_state * gate_i.unsqueeze(-1)  
            cell_state = cell_state + key_value_i 
            
            # Compute output for current timestep - batch matrix multiplication
            output_i = torch.bmm(cell_state.transpose(-2, -1), query_i.unsqueeze(-1)).squeeze(-1)  # (B, value_dim)
            outputs[:, i, :] = output_i

        # Stack outputs at the end
        outputs = self.out_proj(outputs)  # (B, T, output_dim)
            
        return outputs, cell_state

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

    def forward(
            self, 
            input_ids: torch.Tensor,
            cell_state: Optional[torch.Tensor] = None
        ) -> list[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Tensor of shape (B, T) containing token indices

        Returns:
            output: Tensor of shape (B, T, vocab_size) containing token logits
        """
        
        hidden_state = self.embedding(input_ids)  # (B, T, hidden_dim)

        for layer in self.layers:
            hidden_state, cell_state = layer(hidden_state, cell_state)  # Pass through each SimpleRNN layer

        output = self.lm_head(hidden_state)  # (B, T, vocab_size)

        return output, cell_state


def main():
    vocab_size = 512
    hidden_dim = 128
    key_dim = 32
    value_dim = 64
    output_dim = 128
    num_layers = 2

    model = LM(vocab_size, hidden_dim, key_dim, value_dim, output_dim, num_layers)

    print(model)
    
    # Test with batch processing
    batch_size = 4
    seq_length = 10
    
    # Test batched input
    batched_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    batched_output = model(batched_input)
    print(f"Batched input shape: {batched_input.shape}")
    print(f"Batched output shape: {batched_output.shape}")


if __name__ == "__main__":
    main()
