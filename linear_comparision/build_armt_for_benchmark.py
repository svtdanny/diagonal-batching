import torch
import torch.nn as nn
import copy

import sys

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../armt/associative-recurrent-memory-transformer')

from grouped_batching.llama1b_grouping import (
    wrap_model_with_armt
    , get_grouped_states, 
    make_grouped_layer_from_single_layer, make_grouped_model_from_naive,
    make_grouped_sliced_layer_from_single_layer
)
from grouped_batching.fast_executor import (
    FastGroupedArmtExecutor, GroupedLayerContext, 
    associate_with_context, update_mem_with_context
)

from grouped_batching.universal_grouping import (
    extract_params_from_module, get_universal_grouped_states,
    get_module_by_path, make_universal_grouped_layer,
    set_module_by_path, make_universal_grouped_model
)

def preprocess_segment_gpt2(model, input_segment):
    pos_ids = torch.arange(
        0, 
        input_segment.shape[0], 
        device=input_segment.device
    )
    wpe_emb = model.memory_cell.model.transformer.wpe(pos_ids)
    input_segment.add_(wpe_emb)
    return input_segment

grouped_compute_gpt2 = None # uses default grouped layer

def postprocess_segment_gpt2(model, output_segment):
    out = model.memory_cell.model.transformer.ln_f(output_segment)
    out_tokens = model.memory_cell.model.lm_head(out)
    return out_tokens



preprocess_segment_llama = None

def grouped_compute_llama(model, grouped_layer, grouped_input_tensor):
    position_ids = torch.arange(
        0, 
        grouped_input_tensor.shape[1], 
        device=grouped_input_tensor.device
    ).unsqueeze(0)
    position_embeddings = model.memory_cell.model.model.rotary_emb(grouped_input_tensor, position_ids)
    batch_output = grouped_layer.forward(grouped_input_tensor, position_embeddings=position_embeddings)
    return batch_output
    

def postprocess_segment_llama(model, output_segment):
    out = model.memory_cell.model.model.norm(output_segment)
    out_tokens = model.memory_cell.model.lm_head(out)
    return out_tokens

def build_warm_model(model,armt_grouped_model, grouped_layer, grouped_context, model_type, armt_config, dtype):
    ### ONLY FOR FAST LATENCY VERSION

    # compile full layers
    segments_input = torch.rand((model.config.num_hidden_layers, armt_config['segment_size'], model.config.hidden_size), device="cuda", dtype=dtype)

    i, j = 0, model.config.num_hidden_layers
    grouped_context.start_idx = i
    grouped_context.end_idx = j
    grouped_context.is_full = True

    ao = associate_with_context(grouped_layer, grouped_context, segments_input[i:j])
    grouped_layer.generate_mode = True
    if model_type == "gpt2":
        armt_grouped_model.memory_cell.model.transformer(inputs_embeds=segments_input[i:j], use_cache=False)
    else:
        armt_grouped_model.memory_cell.model.model(inputs_embeds=segments_input[i:j], use_cache=False)
    update_mem_with_context(grouped_layer, grouped_context, segments_input[i:j])


def build_armt_for_benchmark(model_name, model, armt_config, dtype, device):
    """
    Build ARMT for benchmark.

    Parameters:
    - model_name: used for understanding the model
    - model: model to build ARMT for

    Returns:
    - model: model wrapped with ARMT
    """
    if "gpt" in model_name:
        MODEL_TYPE = "gpt2"
    elif "llama" in model_name:
        MODEL_TYPE = "llama"
    else:
        raise Exception("Not done yet")
    
    torch.set_default_dtype(dtype)
    torch.set_grad_enabled(False)
    reference_model = copy.deepcopy(model)
    
    if MODEL_TYPE == "gpt2":
        layers_attr = 'transformer.h'
    else:
        layers_attr = 'model.layers'

    torch.manual_seed(0)
    armt_model = wrap_model_with_armt(model, **armt_config, layers_attr=layers_attr)
    armt_model.to(device)
    
    torch.manual_seed(0)
    armt_reference_model = wrap_model_with_armt(reference_model, **armt_config, layers_attr=layers_attr)
    armt_reference_model.to(device)
    
    if MODEL_TYPE == "gpt2":
        grouped_params = get_universal_grouped_states(armt_model.memory_cell.model.transformer.h)
    else:
        grouped_params = get_universal_grouped_states(armt_model.memory_cell.model.model.layers)

    grouped_context = GroupedLayerContext()
    
    if MODEL_TYPE == "gpt2":
        layer_base = copy.deepcopy(armt_model.memory_cell.model.transformer.h[0])
    else:
        layer_base = copy.deepcopy(armt_model.memory_cell.model.model.layers[0])

    grouped_layer = make_universal_grouped_layer(
        grouped_context, 
        layer_base,
        grouped_params,
        use_layer_norm=(MODEL_TYPE == "gpt2"), # layer norm for gpt2, rms norm for llama
    )
    
    from grouped_batching.universal_executor import UniversalGroupedExecutor

    if MODEL_TYPE == "gpt2":
        layers_path = "memory_cell.model.transformer.h"
    else:
        layers_path = "memory_cell.model.model.layers"
    
    armt_grouped_model, source_model_layers = make_universal_grouped_model(
        armt_model, 
        grouped_layer,
        layers_path=layers_path
    )

    if MODEL_TYPE == "gpt2":
        executor = UniversalGroupedExecutor(
            model=armt_grouped_model,
            grouped_layer=grouped_layer,
            context=grouped_context,
            n_layers=model.config.num_hidden_layers,
            model_path="memory_cell.model.transformer",
            out_norm_attr="memory_cell.model.transformer.ln_f",
            lm_head_path="memory_cell.model.lm_head",
            memory_path="memory_cell",
            preprocess_segment_fn = preprocess_segment_gpt2,
            postprocess_segment_fn = postprocess_segment_gpt2,
            grouped_compute_fn = grouped_compute_gpt2,
        )
    else:
        executor = UniversalGroupedExecutor(
            model=armt_grouped_model,
            grouped_layer=grouped_layer,
            context=grouped_context,
            n_layers=model.config.num_hidden_layers,
            model_path="memory_cell.model.model",
            out_norm_attr="memory_cell.model.model.norm",
            lm_head_path="memory_cell.model.lm_head",
            memory_path="memory_cell",
            preprocess_segment_fn = preprocess_segment_llama,
            postprocess_segment_fn = postprocess_segment_llama,
            grouped_compute_fn = grouped_compute_llama,
        )

    executor.vanilla_model = armt_reference_model
    
    build_warm_model(
        model, 
        armt_grouped_model, 
        grouped_layer, 
        grouped_context, 
        MODEL_TYPE, 
        armt_config, 
        dtype
    )
    
    return armt_model, armt_grouped_model, executor
