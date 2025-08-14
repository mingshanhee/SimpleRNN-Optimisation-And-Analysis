import torch
import torch.nn.functional as F


@torch.no_grad()
def simple_inference(model, prefix_tokens, max_new_tokens=100, eos_token_id="</s>", temperature=0.0, top_k=None):
    model.eval()

    generated = []
    for _ in range(max_new_tokens):
        
        # 2) One-step decode: feed last token + cell state
        # print(last_tok.shape, cell_state.shape)
        step_logits = model(prefix_tokens)     # (1, 1, V), cell_state updated
        step_logits = step_logits[-1]  # Get the last timestep logits: (V)

        # 3) Convert to a token (greedy or sample)
        if temperature and temperature > 0:
            probs = F.softmax(step_logits / temperature, dim=-1)
            if top_k is not None and top_k > 0:
                # top-k sampling
                topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                next_tok = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))
            else:
                next_tok = torch.multinomial(probs, num_samples=1)
        else:
            # greedy
            next_tok = step_logits.argmax(dim=-1, keepdim=True)  # (1)

        tok_id = next_tok.item()
        generated.append(tok_id)

        # 4) Stop if EOS
        if tok_id == eos_token_id:
            break

        # 5) Add the next token to the prefix
        prefix_tokens = torch.cat([prefix_tokens, next_tok])
    
    return torch.tensor(generated, dtype=torch.long).unsqueeze(0)  # Return as a 2D tensor

@torch.no_grad()
def modified_inference(
    model,
    prefix_tokens,          # 1D LongTensor, e.g. [BOS, ..., last_prefix_token]
    max_new_tokens=20,
    eos_token_id="</s>",
    temperature=0.0,        # 0.0 => greedy; >0 => sampling
    top_k=None
):
    model.eval()

    # 1) Run the prefix once to get the carried hidden state
    _, cell_state = model(prefix_tokens[:, :-1])  # (1, T-1), h is the hidden state after the prefix

    # start from the *last* token of the prefix
    last_tok = prefix_tokens[:, -1:]

    generated = []
    for _ in range(max_new_tokens):
        
        # 2) One-step decode: feed last token + cell state
        # print(last_tok.shape, cell_state.shape)
        step_logits, cell_state = model(last_tok, cell_state)     # (1, 1, V), cell_state updated
        step_logits = step_logits[:, -1, :]  # Get the last timestep logits: (1, V)

        # 3) Convert to a token (greedy or sample)
        if temperature and temperature > 0:
            probs = F.softmax(step_logits / temperature, dim=-1)
            if top_k is not None and top_k > 0:
                # top-k sampling
                topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                next_tok = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))
            else:
                next_tok = torch.multinomial(probs, num_samples=1)
        else:
            # greedy
            next_tok = step_logits.argmax(dim=-1, keepdim=True)  # (1,1)

        tok_id = next_tok.item()
        generated.append(tok_id)

        # 4) Stop if EOS
        if tok_id == eos_token_id:
            break

        # 5) Feed the token to generate the next one
        last_tok = next_tok

    return torch.tensor(generated, dtype=torch.long).unsqueeze(0)