import torch

def prune_head(attention_module, head_to_prune):
    d_model = attention_module.embed_dim
    num_heads = attention_module.num_heads
    head_dim = d_model // num_heads

    start = head_dim * head_to_prune
    end = start + head_dim

    for proj in [attention_module.q_proj, attention_module.k_proj, attention_module.v_proj]:
        proj.weight.data[:, start:end] = 0
        if proj.bias is not None:
            proj.bias.data[start:end] = 0
    return attention_module

def dynamic_prune(attention_module):
    d_model = attention_module.embed_dim
    num_heads = attention_module.num_heads
    head_dim = d_model // num_heads

    head_norms = torch.zeros(num_heads)
    for proj in [attention_module.q_proj, attention_module.k_proj, attention_module.v_proj]:
        for head in range(num_heads):
            start = head * head_dim
            end = start + head_dim
            head_weights = proj.weight[:, start:end]
            head_norms[head] += head_weights.norm(p=2).item()
    
    head_to_prune = torch.argmin(head_norms).item()
    pruned_attention_module = prune_head(attention_module, head_to_prune)

    return pruned_attention_module
