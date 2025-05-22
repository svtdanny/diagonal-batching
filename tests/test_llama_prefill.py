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
        tC = torch.zeros((x.shape[-2], layer.weight.shape[-2]), device=x.device, dtype=x.dtype)
        tD = torch.zeros((x.shape[-2], layer.weight.shape[-2]), device=x.device, dtype=x.dtype)
        # plan = cutlass.Gemm(
        #     element=x.dtype, 
        #     element_accumulator=torch.float32,
        #     layout_A=cutlass.LayoutType.RowMajor,
        #     layout_B=cutlass.LayoutType.ColumnMajor,
        #     layout_C=cutlass.LayoutType.RowMajor,
        # )
        # plan.run(x, layer.weight.T, tC, tD, print_module=False)
        
        plan = cutlass.GroupedGemm(
            element=x.dtype,
            element_accumulator=torch.float32,
            layout_A=cutlass.LayoutType.RowMajor,
            layout_B=cutlass.LayoutType.ColumnMajor,
            layout_C=cutlass.LayoutType.RowMajor,
        )
        plan.run([x], [layer.weight.T], [tC], [tD], print_module=False)
                
        if layer.bias is not None:
            tD += layer.bias
        if len(x.shape) == 3:
            return tD.unsqueeze(0)
        return tD
    return forward


config = LlamaConfig(
    vocab_size=5000,
    hidden_size=128,
    num_key_value_heads=4,
    num_hidden_layers=8,
    num_attention_heads=4,
    intermediate_size=128,
    hidden_act="silu",
    torch_dtype=dtype,
)

# config = LlamaConfig(
#     vocab_size=5000,
#     hidden_size=2048,
#     num_key_value_heads=32,
#     num_hidden_layers=4,
#     num_attention_heads=32,
#     intermediate_size=8192,
#     hidden_act="silu",
#     torch_dtype=dtype,
# )


armt_config = dict(
    segment_size=128,
    num_mem_tokens=16,
    d_mem=64,
)

num_segments = 100


def test_llama_prefill_correctness():
    test_model = LlamaForCausalLM(config)
    test_model.lm_head = torch.nn.Identity()
    test_model.eval()
    
    reference_model = copy.deepcopy(test_model)

    # for name, module in reference_model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print("MODULE:", name)
            
            
    #         # module.forward = get_manual_linear_forward(module)
    #         # module.forward = get_cublas_linear_forward(module)

    # assert False

    torch.manual_seed(0)
    armt_model = wrap_model_with_armt(test_model, **armt_config)
    armt_model.to("cuda")
    
    torch.manual_seed(0)
    armt_reference_model = wrap_model_with_armt(reference_model, **armt_config)
    armt_reference_model.to("cuda")
    
    # for name, module in armt_reference_model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print("MODULE:", name)
            
    #         if not name.endswith("W_mb"):
    #             module.forward = get_cublas_linear_forward(module)
                
    # assert False
    
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_layer_from_single_layer(
        copy.deepcopy(armt_model.memory_cell.model.model.layers[0]), *grouped_states)
    
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    
    batcher = GroupedBatcher(
        armt_grouped_model, 
        n_layers=config.num_hidden_layers, 
        seg_size=armt_config["segment_size"]+armt_config["num_mem_tokens"], 
        hid_dim=config.hidden_size, 
        pos_embed_dim=config.hidden_size
    )
    executor = ArmtGroupedExecutor(armt_grouped_model, grouped_layer, batcher)
    
    input_ids = torch.randint(0, 5000, (1, num_segments*armt_config["segment_size"]), dtype=torch.long, device="cuda")
    
    reference_output = armt_reference_model.forward(input_ids)
    
    output = executor.forward(input_ids)
    
    print("\nGot output:\n", output.logits)
    print("\nReference output:\n", reference_output.logits)
    
    print(f"output.logits.shape: {output.logits.shape}")
    print(f"reference_output.logits.shape: {reference_output.logits.shape}")
    print("\nDiff between outputs:\n", output.logits - reference_output.logits)
    print("\nDiff between outputs:\n", (output.logits - reference_output.logits).sum(dim=-1).tolist())

    nominator = torch.norm(output.logits - reference_output.logits)
    denominator = torch.norm(reference_output.logits)
    print(f"\nLogits diff: {nominator}/{denominator}={nominator/denominator}")
    
    
if __name__ == "__main__":
    test_llama_prefill_correctness()
