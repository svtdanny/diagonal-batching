import torch
from modeling_amt.language_modeling import AssociativeMemoryCell, AssociativeRecurrentWrapper
from grouped_batching.linear_grouped_training import GroupedLinear


def make_grouped_training_layer_from_single_layer(
    grouped_layer,
    layers
):
    grouped_layer.W_mq = GroupedLinear.from_torch_layers([l.W_mq for l in layers])
    grouped_layer.W_mk = GroupedLinear.from_torch_layers([l.W_mk for l in layers])
    grouped_layer.W_mv = GroupedLinear.from_torch_layers([l.W_mv for l in layers])
    grouped_layer.W_mb = GroupedLinear.from_torch_layers([l.W_mb for l in layers], use_naive_implementation=True)

    W_mem_group = [l.W_mem.data.contiguous() for l in layers]
    z_group = [l.z.data.contiguous() for l in layers]
    grouped_layer.W_mem.data = torch.concat(W_mem_group, dim=0)
    grouped_layer.z.data = torch.concat(z_group, dim=0)

    grouped_layer.layer.self_attn.q_proj = GroupedLinear.from_torch_layers([l.layer.self_attn.q_proj for l in layers])
    grouped_layer.layer.self_attn.k_proj = GroupedLinear.from_torch_layers([l.layer.self_attn.k_proj for l in layers])
    grouped_layer.layer.self_attn.v_proj = GroupedLinear.from_torch_layers([l.layer.self_attn.v_proj for l in layers])
    grouped_layer.layer.self_attn.o_proj = GroupedLinear.from_torch_layers([l.layer.self_attn.o_proj for l in layers])

    grouped_layer.layer.mlp.gate_proj = GroupedLinear.from_torch_layers([l.layer.mlp.gate_proj for l in layers])
    grouped_layer.layer.mlp.up_proj = GroupedLinear.from_torch_layers([l.layer.mlp.up_proj for l in layers])
    grouped_layer.layer.mlp.down_proj = GroupedLinear.from_torch_layers([l.layer.mlp.down_proj for l in layers])

    input_layernorm_group = torch.stack([l.layer.input_layernorm.weight.data for l in layers]).contiguous()
    post_attention_layernorm_group = torch.stack([l.layer.post_attention_layernorm.weight.data for l in layers]).contiguous()
    grouped_layer.layer.input_layernorm.weight.data = input_layernorm_group[:, None, :]
    grouped_layer.layer.post_attention_layernorm.weight.data = post_attention_layernorm_group[:, None, :]

    grouped_layer._grouped_execution = True
    grouped_layer._skip_associating = True

    return grouped_layer
