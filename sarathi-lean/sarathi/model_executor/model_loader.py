"""Utilities for selecting and loading models."""

import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig, GPTQConfig, AutoModelForCausalLM, BitsAndBytesConfig

from sarathi.config import ModelConfig
from sarathi.model_executor.models import *  # pylint: disable=wildcard-import
from sarathi.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "FalconForCausalLM": FalconForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    # "LlamaForCausalLM": AutoModelForCausalLM, # This uses transformers' AutoModelForCausalLM, so it doesn't use the modified EE llama
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "NestedLlamaForCausalLM": NestedLlamaForCausalLM,  # Balcony / multi-exit LLaMA
    "InternLMForCausalLM": InternLMForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "YiForCausalLM": YiForCausalLM,
}

# Models whose HF config reports "LlamaForCausalLM" but require a different class.
_MODEL_NAME_OVERRIDES = {
    "parsakaveh/Balcony-LLaMA2-7B": (NestedLlamaForCausalLM, [15, 18, 21]),
    "parsakaveh/Balcony-LLM-1B": (NestedLlamaForCausalLM, [4, 8, 12]),
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}"
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    if model_config.model in _MODEL_NAME_OVERRIDES:
        model_class, exit_layer_indices = _MODEL_NAME_OVERRIDES[model_config.model]
        model_config.hf_config.exit_layer_indices = exit_layer_indices
        model_config.hf_config.output_exit_layers = exit_layer_indices
        model_config.hf_config.tie_exit_lm_head = True
        model_config.hf_config.exit_decoder_layer = True
    else:
        model_class = _get_model_architecture(model_config.hf_config)
    if model_config.model == '01-ai/Yi-34B':
        model_config.hf_config.hidden_size = 8192
        model_config.hf_config.num_attention_heads = 64
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_config)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
    return model.eval()


"""
In order to load quantized models, we need to use the `from_pretrained` method, but our EE llama doesn't have this method. Only transformers' AutoModelForCausalLM has this method. So we can't use quanted models with our EE llama.
"""
def get_model_quant(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    if model_config.model == '01-ai/Yi-34B':
        model_config.hf_config.hidden_size = 8192
        model_config.hf_config.num_attention_heads = 64
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                #bnb_4bit_quant_type="nf4",    # I've also tried removing this line
                bnb_4bit_compute_dtype=torch.float16,
                #bnb_4bit_use_double_quant=True,    # I've also tried removing this line
            )
            model = model_class.from_pretrained(
                model_config.model,
                config=model_config.hf_config,
                device_map="auto",
                cache_dir=model_config.download_dir,
                # quantization_config=bnb_config,
            )
    return model.eval()


