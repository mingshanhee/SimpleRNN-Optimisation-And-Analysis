import torch
import torch.nn as nn
from typing import Optional, Tuple

# Add gradient checkpointing
from torch.utils.checkpoint import checkpoint


class LuongAttention(nn.Module):
    """
    Global Luong attention over a set of key/value vectors (here we use past outputs as both).
    score 'dot':   score(h_t, h_s) = h_t^T h_s
    score 'general': score(h_t, h_s) = h_t^T W_a h_s
    """
    def __init__(self, dim: int, score: str = "dot"):
        super().__init__()
        assert score in {"dot", "general"}
        self.score = score
        if score == "general":
            self.Wa = nn.Linear(dim, dim, bias=False)  # applied to keys

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, V)       - current step vector (we use your value-space output_i)
            keys:  (B, S, V)    - bank of past step vectors (0..t), causal by construction

        Returns:
            context: (B, V)
            attn_weights: (B, S)
        """
        B, S, V = keys.shape

        # (B, V) -> (B, 1, V) for bmm compatibility
        q = query.unsqueeze(1)  # (B, 1, V)

        if self.score == "dot":
            # scores = keys · q^T -> (B, S, 1) -> (B, S)
            scores = torch.bmm(keys, q.transpose(1, 2)).squeeze(-1)
        else:
            # general: keys' = Wa(keys)
            keys_ = self.Wa(keys)  # (B, S, V)
            scores = torch.bmm(keys_, q.transpose(1, 2)).squeeze(-1)  # (B, S)

        # softmax over S (past steps)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, S)

        # context = sum_s alpha_s * keys_s
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)  # (B, V)

        return context, attn_weights

class SimpleRNN(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            key_dim: int, 
            value_dim: int, 
            output_dim: int,
            fused_projection: bool = True,
            use_luong_attention: bool = False,
            luong_score: str = "general",  # 'dot' or 'general'
            layer_dropout_p: float = 0.1,  # inside-layer dropout (timestep and attention context)
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.fused_projection = fused_projection
        self.use_luong_attention = use_luong_attention

        if fused_projection:
            # Single projection layer for all components
            self.proj = nn.Linear(hidden_dim, 3 * key_dim + value_dim)
        else:
            # Separate projection layers for each component
            self.query_proj = nn.Linear(hidden_dim, key_dim)  # query projection
            self.key_proj = nn.Linear(hidden_dim, key_dim)  # key projection
            self.value_proj = nn.Linear(hidden_dim, value_dim)  # value projection
            self.gate_proj = nn.Linear(hidden_dim, key_dim)  # gate projection

        # self.out_proj = nn.Linear(value_dim, output_dim)  # output projection

        # Attention module (operates in value_dim space)
        if self.use_luong_attention:
            self.attn = LuongAttention(dim=value_dim, score=luong_score)
            # After attention we concatenate [output_i ; context_i] -> 2 * value_dim
            self.out_proj = nn.Linear(2 * value_dim, output_dim)
        else:
            self.out_proj = nn.Linear(value_dim, output_dim)  # output projection (no attention)

        # one dropout module reused inside the layer
        self.layer_dropout = nn.Dropout(layer_dropout_p)

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
            # print("WARNING: cell_state is None, initializing to zeros")
            cell_state = torch.zeros(B, K, V, dtype=dtype, device=device)  # (B, K, V)
        
        # Use list to avoid pre-allocating large tensor
        outputs = torch.empty(B, T, self.output_dim if self.use_luong_attention else V,
                              dtype=dtype, device=device)
        past_values = []  # list of (B, V)
        
        # Process sequence step by step to reduce memory usage
        for i in range(T):
            # Compute projections for current timestep
            h_i = hidden_state[:, i, :]  # (B, hidden_dim)
            
            if self.fused_projection:
                # Single projection for all components
                qkgv = self.proj(h_i)
                query_i, key_i, gate_i, value_i = torch.split(qkgv, [K, K, K, V], dim=-1)
            else:
                query_i = self.query_proj(h_i) 
                key_i = self.key_proj(h_i)  
                gate_i = self.gate_proj(h_i)  
                value_i = self.value_proj(h_i)  

            # activations
            key_i = torch.sigmoid(key_i)  
            gate_i = torch.sigmoid(gate_i) 
            value_i = torch.tanh(value_i)  

            # Compute key-value update - batch matrix multiplication
            key_value_i = torch.bmm(key_i.unsqueeze(-1), value_i.unsqueeze(1))  # (B, key_dim, value_dim)
            
            # Update cell_state (non-in-place to avoid gradient issues)
            cell_state = cell_state * gate_i.unsqueeze(-1)  
            cell_state = cell_state + key_value_i 
            
            # Compute output for current timestep - batch matrix multiplication
            output_i_val = torch.bmm(cell_state.transpose(-2, -1), query_i.unsqueeze(-1)).squeeze(-1)  # (B, value_dim)
            output_i_val = self.layer_dropout(output_i_val)  # inside-layer dropout on timestep output

            if self.use_luong_attention:
                # Build attention keys from past outputs (causal). Include current step for stability.
                past_values.append(output_i_val)
                keys = torch.stack(past_values, dim=1)  # (B, i+1, V)
                context_i, _ = self.attn(output_i_val, keys)  # (B, V), (B, i+1)
                context_i = self.layer_dropout(context_i)         # inside-layer dropout on context

                # Concatenate [output_i ; context_i] -> project to output_dim
                output_i = self.out_proj(torch.cat([output_i_val, context_i], dim=-1))  # (B, output_dim)
            else:
                output_i = self.out_proj(output_i_val)  # (B, V)
                past_values.append(output_i_val)  # still maintain for consistency if needed

            outputs[:, i, :] = output_i

        # # Stack outputs at the end
        # outputs = self.out_proj(outputs)  # (B, T, output_dim)
            
        return outputs, cell_state

class LM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int, 
        key_dim: int, 
        value_dim: int, 
        output_dim: int, 
        num_layers: int,
        fused_projection: bool = True,
        use_gradient_checkpointing: bool = False,
        embed_dropout_p: float = 0.1,         # 1) embedding dropout
        layer_dropout_p: float = 0.1,         # 2) inside-layer dropout (per SimpleRNN)
        inter_layer_dropout_p: float = 0.1,   # 3) between-layer dropout
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
        self.embed_dropout = nn.Dropout(embed_dropout_p)

        # Stack of SimpleRNN layers
        self.layers = nn.ModuleList([
            SimpleRNN(hidden_dim, key_dim, value_dim, output_dim, fused_projection, layer_dropout_p=layer_dropout_p) 
            for _ in range(num_layers)
        ])

        # Between-layer dropout (not applied after the last layer)
        self.inter_layer_dropout = nn.Dropout(inter_layer_dropout_p)

        # Final output projection after reduction
        self.lm_head = nn.Linear(output_dim, vocab_size)

    def _forward_layer(self, layer, hidden_state, cell_state):
        return layer(hidden_state, cell_state)

    def forward(
            self, 
            input_ids: torch.Tensor,
            cell_states: Optional[torch.Tensor] = None
        ) -> list[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Tensor of shape (B, T) containing token indices

        Returns:
            output: Tensor of shape (B, T, vocab_size) containing token logits
        """
        
        hidden_state = self.embedding(input_ids)  # (B, T, hidden_dim)
        hidden_state = self.embed_dropout(hidden_state)

        states = []        
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_state, cell_state = checkpoint(self._forward_layer, layer, hidden_state, cell_states[i] if cell_states is not None else None)
            else:
                hidden_state, cell_state = layer(hidden_state, cell_states[i] if cell_states is not None else None)  # Pass through each SimpleRNN layer

            if i < self.num_layers - 1:  # between-layer dropout only
                hidden_state = self.inter_layer_dropout(hidden_state)

            states.append(cell_state)  # Store cell state for each layer

        output = self.lm_head(hidden_state)  # (B, T, vocab_size)

        return output, torch.stack(states, dim=0)


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
