import torch

def prune_head(attention_module, head_to_prune):
    """
    Prunes a specific attention head in the given attention module.

    This function sets the weights and biases (if present) of the specified attention head to zero.
    This effectively removes the head from contributing to the attention mechanism.

    Args:
        attention_module (torch.nn.Module): The attention module containing the heads.
        head_to_prune (int): The index of the head to prune.

    Returns:
        torch.nn.Module: The modified attention module with the specified head pruned.
    """
    d_model = attention_module.embed_dim
    num_heads = attention_module.num_heads
    head_dim = d_model // num_heads

    start = head_dim * head_to_prune
    end = start + head_dim

    # Iterate over Q, K, V and set the weight data at the head indices to 0
    for proj in [attention_module.q_proj, attention_module.k_proj, attention_module.v_proj]:
        proj.weight.data[:, start:end] = 0
        if proj.bias is not None:
            proj.bias.data[start:end] = 0
    return attention_module

def dynamic_prune(attention_module, num_heads=2):
    """
    Dynamically prunes the least important attention heads from the given attention module.

    This function calculates the norm of the weights for each head in the Q, K, and V projections
    of the attention module. It then prunes the specified number of least important heads based
    on the smallest L2 norms.

    Args:
        attention_module (torch.nn.Module): The attention module containing the heads.
        num_heads (int): The number of heads to prune.

    Returns:
        torch.nn.Module: The modified attention module with the least important heads pruned.
    """
    # Calculate the dimension of each head
    d_model = attention_module.embed_dim
    num_total_heads = attention_module.num_heads
    head_dim = d_model // num_total_heads

    # Initialize tensor to store the norm of each head
    head_norms = torch.zeros(num_total_heads).to(attention_module.device)

    # Calculate the norm for each head across Q, K, V projections
    for proj in [attention_module.q_proj, attention_module.k_proj, attention_module.v_proj]:
        reshaped_weights = proj.weight.data.view(num_total_heads, head_dim, -1)
        head_norms += reshaped_weights.norm(p=2, dim=[1, 2])
    
    # Identify the indices of the heads with the smallest norms
    _, heads_to_prune = torch.topk(head_norms, num_heads, largest=False)

    # Prune the identified heads
    for head in heads_to_prune:
        attention_module = prune_head(attention_module, head.item())

    return attention_module
