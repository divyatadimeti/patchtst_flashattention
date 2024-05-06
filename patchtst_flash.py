import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple
from transformers import PatchTSTForPrediction, PatchTSTConfig
from transformers.models.patchtst.modeling_patchtst import PatchTSTAttention
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.models.llama.modeling_llama import _get_unpad_data

class PatchTSTFlashConfig(PatchTSTConfig):
    """
    PatchTSTFlashConfig extends the PatchTSTConfig with additional configurations specific to the FlashAttention mechanism.

    Attributes:
        causal (bool): If True, the attention mechanism is causal, meaning it prevents the model from attending to future tokens.
        num_key_value_heads (int): The number of key/value heads in the attention mechanism.

    The class constructor accepts additional parameters for causal attention and the number of key/value heads,
    which are specific to the FlashAttention mechanism used in the PatchTST model.
    """
    def __init__(self, 
                 causal,
                 num_key_value_heads,
                 **kwargs
    ):
        super().__init__(**kwargs)
        self.causal = causal
        self.num_key_value_heads = num_key_value_heads

class FlashAttention2(PatchTSTAttention):
    """
    FlashAttention2 extends PatchTSTAttention with FlashAttention mechanism.

    This class implements a variant of the attention mechanism that is optimized for speed and memory efficiency. 

    Attributes:
        causal (bool): If True, the attention mechanism is causal.
        num_key_value_heads (int): The number of key/value heads in the attention mechanism.
    """
    def __init__(self, 
                 causal, 
                 num_key_value_heads,
                 **kwargs):
        super().__init__(**kwargs)
        self.causal = causal
        self.num_key_value_heads = num_key_value_heads
    
    # Code utilized from the Mistral and Phi FlashAttention2 implementation:
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    # Code utilized from the Mistral and Phi FlashAttention2 implementation:
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
    # Code utilized and slightly modified from the Mistral and Phi FlashAttention2 implementation:
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.LongTensor] = None
    ):
        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_dropout = self.dropout if self.training else 0.0

        if query_states.dtype == torch.float32:
            target_dtype = torch.float16

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flashattention_forward(
            query_states, key_states, value_states, attention_mask, query_length, dropout=attn_dropout, softmax_scale=None
        )

        attn_output = attn_output.reshape(batch_size, query_length, -1).contiguous()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # Code utilized and slightly modified from the Mistral and Phi FlashAttention2 implementation:
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi
    def _flashattention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None
    ):
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.causal
            )
        return attn_output

class PatchTSTFlashAttention2(PatchTSTForPrediction):
    """
    PatchTSTFlashAttention2 is a specialized version of the PatchTSTForPrediction class that incorporates
    the FlashAttention2 mechanism into the Transformer encoder layers of the model.

    Attributes:
        config (PatchTSTConfig): Configuration object containing model settings.

    Methods:
        __init__: Initializes the PatchTSTFlashAttention2 model with the specified configuration.
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        for encoder_layer in self.model.encoder.layers:
            encoder_layer.self_attn = FlashAttention2(
                causal=config.causal,
                num_key_value_heads=config.num_key_value_heads,
                embed_dim=config.d_model,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout
            )
