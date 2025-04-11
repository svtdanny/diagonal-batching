from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from grouped_batching.llama1b_grouping import wrap_model_with_armt, get_grouped_states, make_grouped_layer_from_single_layer, make_grouped_model_from_naive
import torch
from grouped_batching.batching import GroupedBatcher
from grouped_batching.executor import ArmtGroupedExecutor

config = LlamaConfig(
    vocab_size=5000,
    hidden_size=128,
    num_key_value_heads=4,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=256,
    hidden_act="silu",
)

armt_config = dict(
    segment_size=128,
    num_mem_tokens=16,
    d_mem=64,
)


def test_llama_grouping_just_works():
    test_model = LlamaForCausalLM(config)
    test_model.eval()
    
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.set_grad_enabled(False)
    
    armt_model = wrap_model_with_armt(test_model, **armt_config)
    armt_model.to("cuda")
    
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_layer_from_single_layer(
        armt_model.memory_cell.model.model.layers[0], *grouped_states)
    
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    
    batcher = GroupedBatcher(
        armt_grouped_model, 
        n_layers=config.num_hidden_layers, 
        seg_size=armt_config["segment_size"]+armt_config["num_mem_tokens"], 
        hid_dim=config.hidden_size, 
        pos_embed_dim=config.hidden_size
    )
    executor = ArmtGroupedExecutor(armt_grouped_model, grouped_layer, batcher)
    
    input_ids = torch.randint(0, 5000, (1, 128), dtype=torch.long, device="cuda")
    output = executor(input_ids)
    print(output)
    
if __name__ == "__main__":
    test_llama_grouping_just_works()
