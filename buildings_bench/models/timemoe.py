# adapter_load_time_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Tuple

import warnings
from typing import Any, Dict, List, Optional, Union

import torch

from transformers import GenerationMixin, LogitsProcessorList, StoppingCriteriaList
from transformers.generation import validate_stopping_criteria, EosTokenCriteria
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.utils import ModelOutput

from typing import List
from transformers import PretrainedConfig

import math
from typing import Optional, Tuple, List, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)


logger = logging.get_logger(__name__)

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass


class TSGenerationMixin(GenerationMixin):

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        input_ids_origin_device = input_ids.device
        input_ids = input_ids.to(self.device)
        if len(input_ids.shape) == 2:
            batch_size, cur_len = input_ids.shape
        else:
            raise ValueError("Input shape must be: [batch_size, seq_len]")
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist()
                for criteria in stopping_criteria
                if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        max_length = stopping_criteria.max_length
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            input_length = input_ids.shape[1]

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                max_horizon_length=max_length - input_length,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens_scores

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            next_tokens = next_tokens.reshape(batch_size, -1, self.config.input_size)
            horizon_length = next_tokens.shape[1]

            input_ids = torch.cat([input_ids, next_tokens], dim=-2)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids[..., 0], scores
            )
            this_peer_finished = unfinished_sequences.max() == 0

        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        if streamer is not None:
            streamer.end()

        input_ids.squeeze_(dim=-1).to(input_ids_origin_device)
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        horizon_length: int = 1,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], horizon_length)
                        ),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], horizon_length)
                        ),
                    ],
                    dim=-1,
                )

        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + horizon_length
            )
            # model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        horizon_lengths: List[int] = 1,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = None,
        hidden_act: str = "silu",
        num_experts_per_tok: int = 2,
        num_experts: int = 1,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        use_dense: bool = False,
        rope_theta: int = 10000,
        attention_dropout: float = 0.0,
        apply_aux_loss: bool = True,
        router_aux_loss_factor: float = 0.02,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = (
            horizon_lengths  # Predict horizon length for each prediction.
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_dense = use_dense
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor

        assert (
            self.use_dense ^ self.apply_aux_loss
        ), "Both use_dense and apply_aux_loss cannot be set to True or False at the same time."

        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
    top_k: int,
    num_experts: int = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if (
        gate_logits is None
        or not isinstance(gate_logits, (tuple, list))
        or gate_logits[0] is None
    ):
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)
    )

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimeMoeInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default 1
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->TimeMOE
class TimeMoeRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->TimeMOE
class TimeMoeRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class TimeMoeMLP(TimeMoeTemporalBlock):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TimeMoeTemporalBlock(
                    hidden_size=self.config.hidden_size,
                    intermediate_size=moe_intermediate_size,
                    hidden_act=self.config.hidden_act,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->TimeMoe
class TimeMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = TimeMoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TimeMoeFlashAttention2(TimeMoeAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:

            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        origin_dtype = query_states.dtype
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            query_states = query_states.to(dtype=torch.bfloat16)
            key_states = key_states.to(dtype=torch.bfloat16)
            value_states = value_states.to(dtype=torch.bfloat16)

        # without attention mask to faster speed
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            return attn_output.to(origin_dtype)
        else:
            return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


TIME_MOE_ATTENTION_CLASSES = {
    "eager": TimeMoeAttention,
    "flash_attention_2": TimeMoeFlashAttention2,
}


class TimeMoeDecoderLayer(nn.Module):
    def __init__(self, config: TimeMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = TIME_MOE_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        if self.config.use_dense:
            self.ffn_layer = TimeMoeMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = TimeMoeSparseExpertsLayer(config)
        self.input_layernorm = TimeMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = TimeMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[torch.FloatTensor],
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value, router_logits


class TimeMoePreTrainedModel(PreTrainedModel):
    config_class = TimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TimeMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TimeMoeModel(TimeMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TimeMoeDecoderLayer`]

    Args:
        config: TimeMoeConfig
    """

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.embed_layer = TimeMoeInputEmbedding(config)
        self.layers = nn.ModuleList(
            [
                TimeMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = TimeMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # input_ids is the input of time series, its shape is [batch_size, seq_len, input_size]
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            if len(input_ids.shape) == 2:
                input_ids.unsqueeze_(dim=-1)
            batch_size, seq_length, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = ()
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            all_router_logits += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class TimeMoeOutputLayer(nn.Module):

    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()

        self.out_layer = nn.Linear(
            hidden_size,
            input_size * horizon_length,
            bias=False,
        )

    def forward(self, x):
        """

            Args:
                x (torch.FloatTensor): with shape [B, seq_len, hidden_size]

            Returns:
        `       torch.FloatTensor: final prediction with shape [B, seq_len, input_size]
        """
        return self.out_layer(x)


class TimeMoeForPrediction(TimeMoePreTrainedModel, TSGenerationMixin):

    def __init__(self, config: TimeMoeConfig):
        super().__init__(config)
        self.config = config
        self.apply_aux_loss = config.apply_aux_loss
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor

        self.model = TimeMoeModel(config)
        # output layer
        lm_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.horizon_lengths):
            lm_head_list.append(
                TimeMoeOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        self.loss_function = torch.nn.HuberLoss(reduction="none", delta=2.0)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        loss_masks: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        max_horizon_length: Optional[int] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        predictions = None

        loss = None
        aux_loss = None
        if labels is not None:
            # AutoRegressive loss
            ar_loss = 0.0
            for lm_head, horizon_length in zip(
                self.lm_heads, self.config.horizon_lengths
            ):
                one_predictions = lm_head(hidden_states)
                one_loss = self.calc_ar_loss(
                    one_predictions, labels, loss_masks, horizon_length
                )
                ar_loss += one_loss
                if predictions is None:
                    predictions = one_predictions
            loss = ar_loss / len(self.config.horizon_lengths)

            if self.apply_aux_loss:
                router_logits = outputs.router_logits if return_dict else outputs[-1]

                temporal_aux_loss = load_balancing_loss_func(
                    router_logits,
                    top_k=self.num_experts_per_tok,
                    num_experts=self.config.num_experts,
                    attention_mask=attention_mask,
                )
                loss += self.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
        else:
            if max_horizon_length is None:
                horizon_length = self.config.horizon_lengths[0]
                max_horizon_length = horizon_length
            else:
                horizon_length = self.config.horizon_lengths[0]
                for h in self.config.horizon_lengths[1:]:
                    if h > max_horizon_length:
                        break
                    else:
                        horizon_length = h
            lm_head = self.lm_heads[self.horizon_length_map[horizon_length]]
            predictions = lm_head(hidden_states)
            if horizon_length > max_horizon_length:
                predictions = predictions[
                    :, :, : self.config.input_size * max_horizon_length
                ]

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss, aux_loss) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_length):
        if len(labels.shape) == 2:
            labels.unsqueeze_(dim=-1)
            # enable model parallelism
            labels = labels.to(predictions.device)
        if loss_masks is not None and len(loss_masks.shape) == 2:
            loss_masks.unsqueeze_(dim=-1)
            # enable model parallelism
            loss_masks = loss_masks.to(predictions.device)

        if horizon_length > 1:
            batch_size, seq_len, output_size = predictions.shape
            shift_predictions = predictions.view(
                batch_size, seq_len, horizon_length, -1
            )

            # pad to the same length with predictions
            # shape -> [B, input_size, seq_len + horizon_length -1]
            labels = F.pad(
                labels.transpose(-1, -2),
                (0, horizon_length - 1),
                mode="constant",
                value=0,
            )

            # shape -> [B, input_size, seq_len, horizon_length]
            shift_labels = labels.unfold(dimension=-1, size=horizon_length, step=1)
            shift_labels = shift_labels.permute(0, 2, 3, 1)

            if loss_masks is not None:
                # pad to the same length with predictions
                loss_masks = F.pad(
                    loss_masks.transpose(-1, -2),
                    (0, horizon_length - 1),
                    mode="constant",
                    value=0,
                )

                loss_masks = loss_masks.unfold(
                    dimension=-1, size=horizon_length, step=1
                )
                loss_masks = loss_masks.permute(0, 2, 3, 1)

        else:
            shift_predictions = predictions
            shift_labels = labels

        # Calculate loss with mask
        losses = self.loss_function(shift_predictions, shift_labels)

        if loss_masks is not None:
            losses = losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = torch.mean(losses)

        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if isinstance(past_key_values, DynamicCache):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length

                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            logger.info("Use input_embedding")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class TimeMoeForecasting(nn.Module):
    """
    外层包装器：
      - 输入：x['load'] (B, ctx+pred, 1)
      - 仅使用过去 ctx 步作为上下文，避免未来信息泄漏
      - 输出：预测未来 pred 步的负荷（与输入同尺度）
      - 将 MoE 的负载均衡辅助损失放到本类的 loss 中统一计算
    """

    def __init__(
        self,
        # 与 LoadForecastingTransformerMoE 对齐的接口参数
        max_context_len: int = 336,
        max_pred_len: int = 168,
        context_len: int = 168,
        pred_len: int = 24,
        # TimeMoe 主干的超参映射
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        num_layers: int = 12,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_act: str = "silu",
        dropout: float = 0.0,  # 目前未直接用到（TimeMoeAttention内部有attention_dropout）
        # 回归头设置
        continuous_loads: bool = True,
        continuous_head: Literal["mse", "gaussian_nll", "huber"] = "mse",
        # MoE 辅助损失权重
        aux_lambda: float = 0.01,
        **kwargs,  # 其他参数（如是否使用 flash attention）
    ):
        super().__init__()
        self.max_context_len = max_context_len
        self.max_pred_len = max_pred_len
        self.default_context_len = context_len
        self.default_pred_len = pred_len

        self.continuous_loads = continuous_loads
        self.continuous_head = continuous_head
        self.aux_lambda = aux_lambda

        # ---- 构造 TimeMoe 主干 ----
        config = TimeMoeConfig(
            input_size=1,
            hidden_size=d_model,
            intermediate_size=dim_feedforward,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            num_key_value_heads=nhead,
            horizon_lengths=[max_pred_len],  # 仅保留最长地平线，外层按需裁剪
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            attention_dropout=dropout,
            use_cache=True,
            use_dense=False,  # 使用稀疏 MoE
            apply_aux_loss=True,  # 需要让路由logits在模型中向外传递
            max_position_embeddings=4096,
        )
        # 设置注意力实现
        try:
            config._attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            )
        except Exception:
            config._attn_implementation = "eager"

        self.backbone = TimeMoeForPrediction(config)

        # ---- 外层预测头（基于最后一个时间步的隐向量）----
        #   - 对于 mse/huber：输出 pred_len * 1
        #   - 对于 gaussian_nll：额外再输出 pred_len * 1 的对数方差（用 softplus^2 转成方差）
        out_dim = self.max_pred_len
        self.mean_head = nn.Linear(d_model, out_dim)
        self.var_head = (
            nn.Linear(d_model, out_dim)
            if (self.continuous_loads and self.continuous_head == "gaussian_nll")
            else None
        )

        # 记录最近一次 forward 的路由 logits 与 mask，用于在 loss 中计算 aux
        self._last_router_logits = None
        self._last_attn_mask = None

    # --------------------- forward ----------------------
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        输入:
            x['load']: (B, ctx+pred, 1)
        输出:
            如果连续回归（mse/huber）：(B, pred_len, 1)
            如果 gaussian_nll：       (B, pred_len, 2)  最后一维 [mu, sigma]
        """
        self._last_router_logits = None
        self._last_attn_mask = None

        if context_len is None:
            context_len = self.default_context_len
        if pred_len is None:
            pred_len = self.default_pred_len

        # 运行时保护：pred_len 不能超过 max_pred_len
        assert (
            pred_len <= self.max_pred_len
        ), f"pred_len={pred_len} 超过 max_pred_len={self.max_pred_len}"

        load = x["load"]  # (B, ctx+pred, 1)
        # 允许 batch 中 pred_len 可变，但至少要有 context_len
        assert load.size(1) >= context_len, "load 长度必须至少包含 context_len"
        # 注意：不再强制等于 ctx+pred（训练脚本如果依赖该校验可保留你原来的逻辑）
        # assert load.size(1) == context_len + pred_len, "load 长度与 ctx+pred 不一致"

        ctx = load[:, :context_len]

        B, _, _ = ctx.shape
        device = ctx.device
        attn_mask = torch.ones(B, context_len, device=device, dtype=torch.long)

        tm_outputs = self.backbone.model(
            input_ids=ctx,
            attention_mask=attn_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        h = tm_outputs.last_hidden_state
        self._last_router_logits = tm_outputs.router_logits
        self._last_attn_mask = attn_mask

        last_h = h[:, -1, :]  # (B, d_model)

        # 先出 max_pred_len，再裁剪到 pred_len
        mu_all = self.mean_head(last_h).reshape(B, self.max_pred_len, 1)
        mu = mu_all[:, :pred_len, :]

        if self.continuous_loads and self.continuous_head == "gaussian_nll":
            logv_all = self.var_head(last_h).reshape(B, self.max_pred_len, 1)
            sigma = F.softplus(logv_all[:, :pred_len, :])
            return torch.cat([mu, sigma], dim=-1)
        else:
            return mu

    # --------------------- loss ---------------------
    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, pred_len, 1) 或 (B, pred_len, 2[mu, sigma])
        y:    (B, pred_len, 1)
        """
        if self.continuous_loads:
            if self.continuous_head == "huber":
                base = F.huber_loss(pred[..., :1], y, delta=1.0, reduction="mean")
            elif self.continuous_head == "mse":
                base = F.mse_loss(pred[..., :1], y, reduction="mean")
            else:  # gaussian_nll
                mu, sigma = pred[..., :1], pred[..., 1:]
                sigma_sq = sigma.pow(2) + 1e-8
                base = 0.5 * (
                    torch.log(2 * torch.pi * sigma_sq) + (y - mu) ** 2 / sigma_sq
                )
                base = base.mean()
        else:
            # 若是分类，可自行更改。这里占位使用 mse
            base = F.mse_loss(pred[..., :1], y, reduction="mean")

        # ---- MoE 负载均衡辅助项（在外层 loss 里加）----
        aux = 0.0
        if self._last_router_logits is not None:
            aux = load_balancing_loss_func(
                gate_logits=tuple(self._last_router_logits),
                top_k=self.backbone.num_experts_per_tok,
                num_experts=self.backbone.config.num_experts,
                attention_mask=self._last_attn_mask,
            )
            aux = aux.to(base.device)

        total = base + self.aux_lambda * aux
        return total

    # --------------------- predict ---------------------
    @torch.no_grad()
    def predict(
        self, x: Dict[str, torch.Tensor], context_len: int = 168, pred_len: int = 24
    ):
        yhat = self.forward(x, context_len, pred_len)
        if self.continuous_loads and self.continuous_head == "gaussian_nll":
            return yhat[..., :1], yhat  # (mu, [mu, sigma])
        else:
            return yhat, yhat

    # ----------------- 其余接口一致 -------------------
    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location="cpu")
        # 常见保存格式兜底
        if (
            isinstance(state, dict)
            and "model" in state
            and isinstance(state["model"], dict)
        ):
            state = state["model"]
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]

        # 去掉 DDP 前缀
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        # 尝试宽松加载（同时覆盖外层和主干）
        missing, unexpected = self.load_state_dict(state, strict=False)
        # 如果主干未被覆盖，再尝试只给主干加载
        if len(missing) > 0 and isinstance(state, dict):
            try:
                _missing, _unexpected = self.backbone.load_state_dict(
                    state, strict=False
                )
            except Exception:
                pass
        return self


def _unwrap_module(m):
    # 兼容 DDP / DataParallel / 你外层的包装
    if hasattr(m, "module"):  # DDP / DP
        return m.module
    return m


def moe_static_params_time_moe(model: nn.Module):
    """
    针对 TimeMoe 模型族（TimeMoeForecasting/TimeMoeForPrediction/TimeMoeModel）的
    一次性静态统计（无需前向）：
      activated_params_total = shared_params + Σ_layer (top_k[layer] * per_expert_params[layer])

    口径：
      - 路由专家（稀疏 experts）只在 top-k 中被使用，因此进入 Σ 项；
      - 其它全部（embedding/attention/norm/MLP、router、shared_expert、输出头等）计入 shared。
    """
    # 1) 拿到实际 backbone，尽量向内解包
    m = _unwrap_module(model)
    # TimeMoeForecasting -> TimeMoeForPrediction -> TimeMoeModel
    if hasattr(m, "backbone"):  # TimeMoeForecasting
        m = _unwrap_module(m.backbone)
    if hasattr(m, "model"):  # TimeMoeForPrediction
        m = _unwrap_module(m.model)

    # 2) 总参数（仅统计 requires_grad=True）
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    # 3) 枚举 MoE 稀疏层（有 .experts 的模块即视为 MoE 层）
    moe_layers = []
    for name, mod in m.named_modules():
        if hasattr(mod, "experts") and isinstance(
            getattr(mod, "experts"), nn.ModuleList
        ):
            moe_layers.append((name or "root", mod))

    # 4) 逐层统计
    per_layer_info = []
    routed_expert_params_total = 0
    activated_expert_params_sum = 0

    for layer_name, mod in moe_layers:
        experts: nn.ModuleList = mod.experts
        n_exp = len(experts)

        # 单专家参数量：用第一个专家求和；若不一致则改用均值
        if n_exp > 0:
            first_sz = sum(
                p.numel() for p in experts[0].parameters() if p.requires_grad
            )
            # 保险起见检查一致性
            sizes = [
                sum(p.numel() for p in e.parameters() if p.requires_grad)
                for e in experts
            ]
            if any(s != first_sz for s in sizes):
                per_exp = int(sum(sizes) // n_exp)
            else:
                per_exp = int(first_sz)
        else:
            per_exp = 0

        # 该层的 top-k（若缺省则尝试从 config 取，最终兜底为 0）
        tk = 0
        if hasattr(mod, "top_k"):
            try:
                tk = int(mod.top_k)
            except Exception:
                tk = 0
        elif hasattr(getattr(mod, "config", None), "num_experts_per_tok"):
            try:
                tk = int(mod.config.num_experts_per_tok)
            except Exception:
                tk = 0

        routed_total_this_layer = per_exp * n_exp
        activated_this_layer = per_exp * tk

        routed_expert_params_total += routed_total_this_layer
        activated_expert_params_sum += activated_this_layer

        per_layer_info.append(
            {
                "layer": layer_name,
                "num_experts": n_exp,
                "per_expert_params": per_exp,
                "top_k": tk,
                "activated_expert_params": activated_this_layer,
                "routed_expert_params_total": routed_total_this_layer,
            }
        )

    # 5) 常驻参数 = 总参数 - 所有层“路由专家”的总和
    shared_params = total_params - routed_expert_params_total
    activated_params_total = shared_params + activated_expert_params_sum

    return {
        "total_params": int(total_params),
        "shared_params": int(shared_params),
        "num_moe_layers": len(moe_layers),
        "activated_expert_params_sum": int(
            activated_expert_params_sum
        ),  # Σ(top_k * per_expert)
        "routed_expert_params_total": int(
            routed_expert_params_total
        ),  # 仅统计路由专家（不含 shared_expert）
        "activated_params_total": int(
            activated_params_total
        ),  # shared + Σ(top_k * per_expert)
        "activation_rate": float(activated_params_total / max(1, total_params)),
        "layers": per_layer_info,  # 分层明细
    }


# ======================= Quick self-test for the wrapper =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TimeMoeForecasting().to(device).train()

    # model参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)

    moe_static = moe_static_params_time_moe(model)

    print("MoE static parameters:", moe_static)

    ctx, pred = 96, 24
    B = 2
    # 伪造 batch（与上游保持一致：只需 load；stl_* 在此包装器中不使用）
    load = torch.rand(B, ctx + pred, 1, device=device)
    dummy = {"load": load}
    target = dummy["load"][:, -pred:]  # 未来真实值（同尺度）

    out = model(dummy, context_len=ctx, pred_len=pred)
    loss = model.loss(out, target)
    loss.backward()
    print("Sanity-check OK – loss:", float(loss))

    model.eval()
    preds, _ = model.predict(dummy, context_len=ctx, pred_len=pred)
    print(
        "Inference preds shape:", preds.shape
    )  # [B, pred, 1] 或 [B, pred, 1] (gaussian取 mu)
