import sys
sys.path.append("./associative-recurrent-memory-transformer")
sys.path.append("..")

import torch
import torch.nn as nn
from llama1b_grouping import get_llama1b_model, wrap_model_with_armt, get_grouped_states, make_grouped_layer_from_single_layer, make_grouped_model_from_naive
from batching import GroupedBatcher
from executor import ArmtGroupedExecutor
import time

def one_simple_forward_main():
    torch.set_grad_enabled(False)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    
    armt_model = get_llama1b_model(dtype)
    armt_model.lm_head = nn.Identity()
    armt_model.eval()
    
    armt_model = wrap_model_with_armt(armt_model, segment_size=512, num_mem_tokens=128, d_mem=64)
    armt_model.memory_cell.model.lm_head = torch.nn.Identity()
    armt_model.to("cuda")
    
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_layer_from_single_layer(
        armt_model.memory_cell.model.model.layers[0], *grouped_states)
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    
    batcher = GroupedBatcher(armt_grouped_model, n_layers=16, seg_size=512+128, hid_dim=2048, pos_embed_dim=2048)
    executor = ArmtGroupedExecutor(armt_grouped_model, grouped_layer, batcher)

    seq_len = 1024*2
    input_ids=torch.randint(0, 10000, (1, seq_len), dtype=torch.long, device="cuda")

    start_time = time.time()
    
    with torch.no_grad():
        o0 = executor.forward(input_ids)

    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
    # print(o0)
    
if __name__ == "__main__":
    one_simple_forward_main()
