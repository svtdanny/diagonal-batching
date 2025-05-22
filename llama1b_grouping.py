import torch
import transformers
from transformers import AutoModelForCausalLM
from modeling_amt.language_modeling_old import AssociativeMemoryCell, AssociativeRecurrentWrapper
from grouped_batching.linear_grouped_forward import get_grouped_gemm_forward, get_naive_grouped_forward
from grouped_batching.linear_grouped_sliced_forward import get_grouped_gemm_sliced_forward, get_naive_grouped_sliced_forward, get_sliced_rms_norm_forward

def get_llama1b_model(dtype):
    source_model_dualed = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"
                                             , attn_implementation="sdpa"
                                             ,torch_dtype=dtype)
    return source_model_dualed

def wrap_model_with_armt(source_model, segment_size, num_mem_tokens, d_mem=64, layers_attr='model.layers'):
    mem_cell_cls = AssociativeMemoryCell
    rec_wrap_cls = AssociativeRecurrentWrapper

    mem_cell_args = dict(
            base_model=source_model,
            num_mem_tokens=num_mem_tokens,
    )
    if d_mem is not None:
        mem_cell_args['d_mem'] = d_mem


    cell = mem_cell_cls(**mem_cell_args, wrap_pos=False, layers_attr=layers_attr)
    armt_model = rec_wrap_cls(cell, segment_size=segment_size, k2=-1)
    
    return armt_model

def get_grouped_states(armt_model):
    W_mq_group = [l.W_mq.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mk_group = [l.W_mk.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mv_group = [l.W_mv.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mb_group = [l.W_mb.weight.data.T.contiguous() for l in armt_model.memory_cell.model.model.layers]
    W_mb_bias_group = [l.W_mb.bias.data.contiguous() for l in armt_model.memory_cell.model.model.layers]
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
        W_mq_group, W_mk_group, W_mv_group, W_mb_group, W_mb_bias_group,
        W_mem_group, z_group, 
        q_proj_group, k_proj_group, v_proj_group, o_proj_group, 
        gate_proj_group, up_proj_group, down_proj_group, 
        input_layernorm_group, post_attention_layernorm_group
    )


def make_grouped_layer_from_single_layer(
    grouped_layer,
    W_mq_group, W_mk_group, W_mv_group, W_mb_group, W_mb_bias_group,
    W_mem_group, z_group, 
    q_proj_group, k_proj_group, v_proj_group, o_proj_group, 
    gate_proj_group, up_proj_group, down_proj_group, 
    input_layernorm_group, post_attention_layernorm_group,
    device='cuda'
    ):
    grouped_layer.W_mq.forward = get_grouped_gemm_forward(W_mq_group)
    grouped_layer.W_mk.forward = get_grouped_gemm_forward(W_mk_group)
    grouped_layer.W_mv.forward = get_grouped_gemm_forward(W_mv_group)
    grouped_layer.W_mb.forward = get_naive_grouped_forward(W_mb_group, torch.stack(W_mb_bias_group)[..., None].to(device))

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

    grouped_layer._grouped_execution = True
    grouped_layer._skip_associating = True

    return grouped_layer

def make_grouped_model_from_naive(armt_model, grouped_layer):
    source_model_layers = armt_model.memory_cell.model.model.layers
    armt_model.out_norm = armt_model.memory_cell.model.model.norm
    armt_model.memory_cell.model.model.norm = torch.nn.Identity()

    armt_model.memory_cell.model.model.layers = torch.nn.ModuleList(
        [grouped_layer]
    )
    
    return armt_model, source_model_layers


def make_grouped_sliced_layer_from_single_layer(
    context,
    grouped_layer,
    W_mq_group, W_mk_group, W_mv_group, W_mb_group, W_mb_bias_group,
    W_mem_group, z_group, 
    q_proj_group, k_proj_group, v_proj_group, o_proj_group, 
    gate_proj_group, up_proj_group, down_proj_group, 
    input_layernorm_group, post_attention_layernorm_group,
    device='cuda',
    grouped_fn = get_grouped_gemm_sliced_forward
):
    grouped_layer.W_mq.forward = grouped_fn(context, W_mq_group)
    grouped_layer.W_mk.forward = grouped_fn(context, W_mk_group)
    grouped_layer.W_mv.forward = grouped_fn(context, W_mv_group)
    grouped_layer.W_mb.forward = get_naive_grouped_sliced_forward(context, W_mb_group, torch.stack(W_mb_bias_group)[..., None].to(device))

    grouped_layer.W_mem.data = torch.concat(W_mem_group, dim=0).to(device)
    grouped_layer.z.data = torch.concat(z_group, dim=0).to(device)

    grouped_layer.layer.self_attn.q_proj.forward = grouped_fn(context, q_proj_group)
    grouped_layer.layer.self_attn.k_proj.forward = grouped_fn(context, k_proj_group)
    grouped_layer.layer.self_attn.v_proj.forward = grouped_fn(context, v_proj_group)
    grouped_layer.layer.self_attn.o_proj.forward = grouped_fn(context, o_proj_group)


    grouped_layer.layer.mlp.gate_proj.forward = grouped_fn(context, gate_proj_group)
    grouped_layer.layer.mlp.up_proj.forward = grouped_fn(context, up_proj_group)
    grouped_layer.layer.mlp.down_proj.forward = grouped_fn(context, down_proj_group)


    grouped_layer.layer.input_layernorm.weight.data = input_layernorm_group[:, None, :]
    grouped_layer.layer.post_attention_layernorm.weight.data = post_attention_layernorm_group[:, None, :]
    grouped_layer.layer.input_layernorm.forward = get_sliced_rms_norm_forward(grouped_layer.layer.input_layernorm, context)
    grouped_layer.layer.post_attention_layernorm.forward = get_sliced_rms_norm_forward(grouped_layer.layer.post_attention_layernorm, context)
    
    grouped_layer._grouped_execution = True
    grouped_layer._skip_associating = True

    return grouped_layer

def make_naive_model_from_grouped(armt_model, grouped_layer):
    """Simply patch all layers from grouped_layer back into original ARMT"""
    def transform_sliced_layer(orig, sliced, idx):
        orig.weight.data.copy_(sliced.wg[idx].data.T)
        if orig.bias is not None:
            orig.bias.data.copy_(sliced.bias[idx][0].data)
    for idx, l in enumerate(armt_model.memory_cell.model.model.layers):
        # copy W_* weights
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].W_mq, grouped_layer.W_mq, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].W_mk, grouped_layer.W_mk, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].W_mv, grouped_layer.W_mv, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].W_mb, grouped_layer.W_mb, idx)
        # mem weights
        armt_model.memory_cell.model.model.layers[idx].W_mem.data.copy_(grouped_layer.W_mem[idx].data)
        armt_model.memory_cell.model.model.layers[idx].z.data.copy_(grouped_layer.z[idx].data)
        # attn
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.self_attn.q_proj,
                               grouped_layer.layer.self_attn.q_proj, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.self_attn.k_proj,
                               grouped_layer.layer.self_attn.k_proj, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.self_attn.v_proj,
                               grouped_layer.layer.self_attn.v_proj, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.self_attn.o_proj,
                               grouped_layer.layer.self_attn.o_proj, idx)
        # mlp
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.mlp.gate_proj,
                               grouped_layer.layer.mlp.gate_proj, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.mlp.up_proj,
                               grouped_layer.layer.mlp.up_proj, idx)
        transform_sliced_layer(armt_model.memory_cell.model.model.layers[idx].layer.mlp.down_proj,
                               grouped_layer.layer.mlp.down_proj, idx)
        # layer norms
        armt_model.memory_cell.model.model.layers[idx].layer.input_layernorm.weight.data.copy_(grouped_layer.layer.input_layernorm.weight[idx][0].data.T)
        armt_model.memory_cell.model.model.layers[idx].layer.post_attention_layernorm.weight.data.copy_(grouped_layer.layer.post_attention_layernorm.weight[idx][0].data.T)
    return armt_model
