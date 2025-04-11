import torch
import transformers
from transformers import AutoModelForCausalLM
from modeling_amt.language_modeling import AssociativeMemoryCell, AssociativeRecurrentWrapper
from grouped_batching.linear_grouped_forward import get_grouped_gemm_forward, get_naive_grouped_forward

def get_llama1b_model(dtype):
    source_model_dualed = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"
                                             , attn_implementation="sdpa"
                                             ,torch_dtype=dtype)
    return source_model_dualed

def wrap_model_with_armt(source_model, segment_size, num_mem_tokens, d_mem=64):
    mem_cell_cls = AssociativeMemoryCell
    rec_wrap_cls = AssociativeRecurrentWrapper

    mem_cell_args = dict(
            base_model=source_model,
            num_mem_tokens=num_mem_tokens,
    )
    if d_mem is not None:
        mem_cell_args['d_mem'] = d_mem


    cell = mem_cell_cls(**mem_cell_args, wrap_pos=False, layers_attr="model.layers")
    armt_model = rec_wrap_cls(cell, segment_size=segment_size, k2=-1)
    
    return armt_model

def get_grouped_states(armt_model):
    W_mq_group = [l.W_mq.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mk_group = [l.W_mk.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mv_group = [l.W_mv.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mb_group = [l.W_mb.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]

    W_mem_group = [l.W_mem.data.contiguous() for l in armt_model.memory_cell.model.model.layers]
    z_group = [l.z.data.contiguous() for l in armt_model.memory_cell.model.model.layers]

    q_proj_group = [l.layer.self_attn.q_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    k_proj_group = [l.layer.self_attn.k_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    v_proj_group = [l.layer.self_attn.v_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    o_proj_group = [l.layer.self_attn.o_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]

    gate_proj_group = [l.layer.mlp.gate_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    up_proj_group = [l.layer.mlp.up_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    down_proj_group = [l.layer.mlp.down_proj.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]

    input_layernorm_group = [l.layer.input_layernorm.weight.data for l in armt_model.memory_cell.model.model.layers]
    post_attention_layernorm_group = [l.layer.post_attention_layernorm.weight.data for l in armt_model.memory_cell.model.model.layers]

    input_layernorm_group = torch.stack(input_layernorm_group).contiguous()
    post_attention_layernorm_group = torch.stack(post_attention_layernorm_group).contiguous()
    
    return (
        W_mq_group, W_mk_group, W_mv_group, W_mb_group, 
        W_mem_group, z_group, 
        q_proj_group, k_proj_group, v_proj_group, o_proj_group, 
        gate_proj_group, up_proj_group, down_proj_group, 
        input_layernorm_group, post_attention_layernorm_group
    )


def make_grouped_layer_from_single_layer(
    grouped_layer,
    W_mq_group, W_mk_group, W_mv_group, W_mb_group, 
    W_mem_group, z_group, 
    q_proj_group, k_proj_group, v_proj_group, o_proj_group, 
    gate_proj_group, up_proj_group, down_proj_group, 
    input_layernorm_group, post_attention_layernorm_group,
    device='cuda'
    ):
    grouped_layer.W_mq.forward = get_grouped_gemm_forward(W_mq_group)
    grouped_layer.W_mk.forward = get_grouped_gemm_forward(W_mk_group)
    grouped_layer.W_mv.forward = get_grouped_gemm_forward(W_mv_group)
    grouped_layer.W_mb.forward = get_naive_grouped_forward(W_mb_group)

    grouped_layer.W_mem.data = torch.concat(W_mem_group, dim=0).to(device)
    grouped_layer.z.data = torch.concat(z_group, dim=0).to(device)

    grouped_layer.layer.self_attn.q_proj.forward = get_grouped_gemm_forward(q_proj_group)
    grouped_layer.layer.self_attn.k_proj.forward = get_grouped_gemm_forward(k_proj_group)
    grouped_layer.layer.self_attn.v_proj.forward = get_grouped_gemm_forward(v_proj_group)
    grouped_layer.layer.self_attn.o_proj.forward = get_grouped_gemm_forward(o_proj_group)


    grouped_layer.layer.mlp.gate_proj.forward = get_grouped_gemm_forward(gate_proj_group)
    grouped_layer.layer.mlp.up_proj.forward = get_grouped_gemm_forward(up_proj_group)
    grouped_layer.layer.mlp.down_proj.forward = get_grouped_gemm_forward(down_proj_group)


    grouped_layer.layer.input_layernorm.weight.data = input_layernorm_group[:, None, :]
    grouped_layer.layer.post_attention_layernorm.weight.data = post_attention_layernorm_group[:, None, :]

    return grouped_layer

def make_grouped_model_from_naive(armt_model, grouped_layer):
    source_model_layers = armt_model.memory_cell.model.model.layers
    armt_model.out_norm = armt_model.memory_cell.model.model.norm
    armt_model.memory_cell.model.model.norm = torch.nn.Identity()

    armt_model.memory_cell.model.model.layers = torch.nn.ModuleList(
        [grouped_layer]
    )
    
    return armt_model, source_model_layers

