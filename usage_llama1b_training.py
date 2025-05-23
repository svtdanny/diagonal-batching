import sys

sys.path.append("./associative-recurrent-memory-transformer")
sys.path.append("..")

import copy
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from grouped_batching.llama1b_grouping import (
    wrap_model_with_armt, get_grouped_states, 
    make_grouped_layer_from_single_layer, make_grouped_model_from_naive,
    make_grouped_sliced_layer_from_single_layer
)
from grouped_batching.batching import GroupedBatcher
from grouped_batching.executor import ArmtGroupedExecutor
from grouped_batching.fast_executor import FastGroupedArmtExecutor, GroupedLayerContext, associate_with_context, update_mem_with_context

torch.autograd.set_detect_anomaly(True)

dtype = torch.bfloat16
torch.set_default_dtype(dtype)
# torch.set_grad_enabled(False)

source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B"
                                            #  , attn_implementation="sdpa"
                                            , attn_implementation="flash_attention_2"
                                             ,torch_dtype=dtype)
source_model.eval()
source_model.lm_head = torch.nn.Identity()
reference_model = copy.deepcopy(source_model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

model_config = source_model.config
armt_config = dict(
    # segment_size=32,
    # num_mem_tokens=16,
    # segment_size=512,
    # num_mem_tokens=128,
    segment_size=1024,
    num_mem_tokens=128,
    d_mem=64,
)

torch.manual_seed(0)
armt_model = wrap_model_with_armt(source_model, **armt_config)
armt_model.to("cuda")

torch.manual_seed(0)
armt_reference_model = wrap_model_with_armt(reference_model, **armt_config)
armt_reference_model.to("cuda")


from grouped_batching.llama1b_grouping_autograd import make_grouped_training_layer_from_single_layer, make_grouped_sliced_training_layer_from_single_layer

### FAST LATENCY TRAINING VERSION

grouped_context = GroupedLayerContext()
grouped_context.is_training = True

# grouped_states = get_grouped_states(armt_model)
grouped_layer = make_grouped_sliced_training_layer_from_single_layer(
    grouped_context, copy.deepcopy(armt_model.memory_cell.model.model.layers[0]), armt_model.memory_cell.model.model.layers
)
armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)


executor = FastGroupedArmtExecutor(
    armt_grouped_model, 
    grouped_layer, 
    grouped_context, 
    model_config.num_hidden_layers, 
)

### ONLY FOR FAST LATENCY VERSION

# compile full layers
segments_input = torch.rand((model_config.num_hidden_layers, armt_config["segment_size"], 2048), device="cuda", dtype=dtype)

i, j = 0, 16
grouped_context.start_idx = i
grouped_context.end_idx = j
grouped_context.is_full = True

ao = associate_with_context(grouped_layer, grouped_context, segments_input[i:j])
grouped_layer.generate_mode = True
armt_grouped_model.memory_cell.model.model(inputs_embeds=segments_input[i:j], use_cache=False)
update_mem_with_context(grouped_layer, grouped_context, segments_input[i:j])


# Actual training

# 4096, 8192, 16384, 32768, 65536, 131072
# num_segments = 4096//armt_config["segment_size"]
num_segments = 4
input_ids = torch.randint(
    0, 10000, 
    (1, num_segments*armt_config["segment_size"]), 
    dtype=torch.long, 
    device="cuda"
)

armt_reference_model.memory_cell.generate_mode(False)
reference_output = armt_reference_model.forward(input_ids)

torch.cuda.synchronize()

print("ACTUAL FORWARD")

for i in range(10):
    print(f"ITER {i} " + "!"*10 )
    executor.armt_model.zero_grad()
    
    output = executor.forward(input_ids)
    # output.logits.sum().backward(retain_graph=True)
    output.logits.sum().backward()