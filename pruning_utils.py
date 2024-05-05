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

def dynamic_prune(attention_module, num_heads=1):
    d_model = attention_module.embed_dim
    num_heads = attention_module.num_heads
    head_dim = d_model // num_heads

    head_norms = torch.zeros(num_heads)
    for proj in [attention_module.q_proj, attention_module.k_proj, attention_module.v_proj]:
        reshaped_weights = proj.weight.data.view(num_heads, head_dim, -1)
        head_norms += reshaped_weights.norm(p=2, dim=[1, 2])
    
    _, heads_to_prune = torch.topk(head_norms, num_heads, largest=False)

    for head in heads_to_prune:
        pruned_attention_module = prune_head(attention_module, head)

    return pruned_attention_module
