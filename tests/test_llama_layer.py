from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from grouped_batching.llama1b_grouping import wrap_model_with_armt, get_grouped_states, make_grouped_layer_from_single_layer, make_grouped_model_from_naive
import torch
from grouped_batching.batching import GroupedBatcher
from grouped_batching.executor import ArmtGroupedExecutor
import copy
import cutlass

dtype = torch.bfloat16
torch.set_default_dtype(dtype)
torch.set_grad_enabled(False)

def get_manual_linear_forward(layer):
    def forward(x):
        # print(f"{x} @ {layer.weight.T} + {layer.bias} = ...")
        o = x @ layer.weight.T
        if layer.bias is not None:
            o += layer.bias
        return o
    return forward

def get_cublas_linear_forward(layer):
    def forward(x):
        # print(f"{x} @ {layer.weight.T} + {layer.bias} = ...")
        tC = torch.zeros(x.shape[:-1] + (layer.weight.shape[-2],), device=x.device, dtype=x.dtype)
        tD = torch.zeros(x.shape[:-1] + (layer.weight.shape[-2],), device=x.device, dtype=x.dtype)
        plan = cutlass.Gemm(
            element=x.dtype, 
            element_accumulator=torch.float32,
            layout_A=cutlass.LayoutType.RowMajor,
            layout_B=cutlass.LayoutType.ColumnMajor,
            layout_C=cutlass.LayoutType.RowMajor,
        )
        plan.run(x, layer.weight.T, tC, tD, print_module=False)
                
        if layer.bias is not None:
            tD += layer.bias
        return tD
    return forward

config = LlamaConfig(
    vocab_size=5000,
    hidden_size=2048,
    num_key_value_heads=8,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=4096,
    hidden_act="silu",
    torch_dtype=dtype,
)

armt_config = dict(
    segment_size=256,
    num_mem_tokens=64,
    # segment_size=32,
    # num_mem_tokens=16,
    d_mem=64,
)

def get_test_layer_input(config, W_shape, z_shape):
    W_batch = torch.rand((config.num_hidden_layers,) + W_shape[1:], device="cuda", dtype=dtype)
    z_batch = torch.rand((config.num_hidden_layers,) + z_shape[1:], device="cuda", dtype=dtype)
    
    segment_batch = torch.rand(
        (config.num_hidden_layers, armt_config["segment_size"]+armt_config["num_mem_tokens"], config.hidden_size),
        device="cuda",
        dtype=dtype
    )

    position_ids_batch = torch.rand(
        (config.num_hidden_layers, armt_config["segment_size"]+armt_config["num_mem_tokens"]),
        device="cuda",
        dtype=dtype
    )
    
    return segment_batch, position_ids_batch, W_batch, z_batch

def test_llama_layers_correctness():
    test_model = LlamaForCausalLM(config)
    test_model.eval()

    armt_model = wrap_model_with_armt(test_model, **armt_config)
    armt_model.to("cuda")
    
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_layer_from_single_layer(
        copy.deepcopy(armt_model.memory_cell.model.model.layers[0]), *grouped_states)
    
    # for name, module in armt_model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         # print(name, type(module))
    #         if "W_mb" in name:
    #             print(f"Skipping {name}")
    #             module.forward = get_manual_linear_forward(module)
    #         else:
    #             # module.forward = get_manual_linear_forward(module)
    #             module.forward = get_cublas_linear_forward(module)
    
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    
    W_shape = source_model_layers[0].W_mem.shape
    z_shape = source_model_layers[0].z.shape
    
    segment_batch, position_ids_batch, W_batch, z_batch = get_test_layer_input(config, W_shape, z_shape)
    
    print("Test grouped associate")
    for l in range(config.num_hidden_layers):
        W_e = W_batch[l:l+1]
        z_e = z_batch[l:l+1]
        
        grouped_layer.W_mem.data.copy_(W_batch)
        grouped_layer.z.data.copy_(z_batch)
        
        s = segment_batch[l:l+1]
        
        source_model_layers[l].W_mem.copy_(W_e)
        source_model_layers[l].z.copy_(z_e)
        a_n = source_model_layers[l].associate(s)
        a_g = grouped_layer.associate(segment_batch)
        
        norm_nominator = torch.norm(a_n - a_g[l])
        norm_denominator = torch.norm(a_n)
        print(f"Associate diff layer {l}: {norm_nominator}/{norm_denominator}={norm_nominator/norm_denominator}")
        # assert norm_nominator/norm_denominator < 1e-3
    
    
    print("Test memory update")
    mem_tokens_batch = torch.rand((config.num_hidden_layers, armt_config["num_mem_tokens"], config.hidden_size), device="cuda", dtype=dtype)
    is_fs = False
    # grouped_layer.first_seg = is_fs
    if is_fs:
        grouped_layer._first_seg_mask = torch.ones(config.num_hidden_layers, device="cuda", dtype=torch.bool)
    else:
        grouped_layer._first_seg_mask = torch.zeros(config.num_hidden_layers, device="cuda", dtype=torch.bool)
        
    
    for l in range(config.num_hidden_layers):
        mem_tokens = mem_tokens_batch[l:l+1]
        
        source_model_layers[l].W_mem.copy_(W_batch[l:l+1])
        source_model_layers[l].z.copy_(z_batch[l:l+1])

        grouped_layer.W_mem.data.copy_(W_batch)
        grouped_layer.z.data.copy_(z_batch)
        
        
        grouped_layer.update_mem(mem_tokens_batch)
        source_model_layers[l].first_seg = is_fs
        source_model_layers[l].update_mem(mem_tokens)
        
        W_g = grouped_layer.W_mem[l]
        z_g = grouped_layer.z[l]
        
        W_n = source_model_layers[l].W_mem
        z_n = source_model_layers[l].z
        
        norm_nominator = torch.norm(W_g - W_n)
        norm_denominator = torch.norm(W_g)
        print(f"Associate diff layer W_mem {l}: {norm_nominator}/{norm_denominator}={norm_nominator/norm_denominator}")
        # assert norm_nominator/norm_denominator < 1e-3
        
        norm_nominator = torch.norm(z_g - z_n)
        norm_denominator = torch.norm(z_g)
        print(f"Associate diff layer z {l}: {norm_nominator}/{norm_denominator}={norm_nominator/norm_denominator}")
        # assert norm_nominator/norm_denominator < 1e-3
        

    print("Test forward for whole layer")
    is_fs = False
    is_gm = False
    
    grouped_layer.first_seg = is_fs
    grouped_layer.generate_mode = is_gm
    if is_fs:
        grouped_layer._first_seg_mask = torch.ones(config.num_hidden_layers, device="cuda", dtype=torch.bool)
    else:
        grouped_layer._first_seg_mask = torch.zeros(config.num_hidden_layers, device="cuda", dtype=torch.bool)
    
    o_g = grouped_layer.forward(segment_batch, position_ids=position_ids_batch)
    
    for l in range(config.num_hidden_layers):
        s = segment_batch[l:l+1]
        p = position_ids_batch[l:l+1]
        
        grouped_layer.W_mem.data.copy_(W_batch)
        grouped_layer.z.data.copy_(z_batch)
        
        source_model_layers[l].W_mem.copy_(W_batch[l:l+1])
        source_model_layers[l].z.copy_(z_batch[l:l+1])
        
        for name, module in source_model_layers[l].layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                # print(name, type(module))
                if "W_mb" in name:
                    print(f"Skipping {name}")
                    module.forward = get_manual_linear_forward(module)
                else:
                    # module.forward = get_manual_linear_forward(module)
                    module.forward = get_cublas_linear_forward(module)

        
        source_model_layers[l].first_seg = is_fs
        source_model_layers[l].generate_mode = is_gm
        o_n = source_model_layers[l].layer.forward(s, position_ids=p)
        norm_nominator = torch.norm(o_g[0][l] - o_n[0])
        norm_denominator = torch.norm(o_g[0][l])
        print(f"Forward diff layer {l}: {norm_nominator}/{norm_denominator}={norm_nominator/norm_denominator}")
        # assert norm_nominator/norm_denominator < 1e-3
    
    
    
if __name__ == "__main__":
    test_llama_layers_correctness()
