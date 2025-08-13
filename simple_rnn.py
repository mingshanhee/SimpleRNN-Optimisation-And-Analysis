import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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

        query: torch.Tensor = self.query_proj(hidden_state)  # (T, key_dim)
        key: torch.Tensor = F.sigmoid(self.key_proj(hidden_state))  # (T, key_dim)
        gate: torch.Tensor = F.sigmoid(self.gate_proj(hidden_state))  # (T, key_dim)
        value: torch.Tensor = self.value_proj(hidden_state)  # (T, value_dim)

        state = torch.zeros(K, V, dtype=dtype, device=device)
        output = torch.zeros(T, V, dtype=dtype, device=device)

        for i in range(T):
            query_i = query[i]  # (key_dim)
            key_i = key[i]  # (key_dim)
            value_i = value[i]  # (value_dim)
            gate_i = gate[i]  # (key_dim)

            key_value_i = key_i.unsqueeze(-1) * value_i.unsqueeze(0)  # (key_dim, value_dim)
            state = state * gate_i.unsqueeze(-1) + key_value_i  # (key_dim, value_dim)
            output[i] = (query_i.unsqueeze(-1) * state).sum(0)  # (value_dim)

        return self.out_proj(output), state


class LM(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dim: int, key_dim: int, value_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Embedding layer to convert input_ids to hidden states
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Stack of SimpleRNN layers
        self.layers = nn.ModuleList([SimpleRNN(hidden_dim, key_dim, value_dim, output_dim) for _ in range(num_layers)])

        # Final output projection after reduction
        self.lm_head = nn.Linear(output_dim, vocab_size)  # Output logits for the vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (T,) containing token indices

        Returns:
            output: Tensor of shape (T, vocab_size) containing token logits
        """
        T = input_ids.shape[0]

        hidden_state = self.embedding(input_ids)  # (T, hidden_dim)

        for layer in self.layers:
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
