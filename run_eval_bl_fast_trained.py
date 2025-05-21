import sys
sys.path.append("./associative-recurrent-memory-transformer")
sys.path.append("./")
sys.path.append("../")
import cutlass
import time
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grouped_batching.llama1b_grouping import wrap_model_with_armt, get_grouped_states, make_grouped_layer_from_single_layer, make_grouped_model_from_naive, make_grouped_sliced_layer_from_single_layer
from grouped_batching.batching import GroupedBatcher
from grouped_batching.executor import ArmtGroupedExecutor
from grouped_batching.fast_executor import FastGroupedArmtExecutor, GroupedLayerContext, associate_with_context, update_mem_with_context
from grouped_batching.llama1b_grouping_autograd import make_grouped_training_layer_from_single_layer, make_grouped_sliced_training_layer_from_single_layer
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import save_model
from modeling_amt.language_modeling import AssociativeRecurrentWrapper, AssociativeMemoryCell
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from tqdm import tqdm
import datasets
from pathlib import Path
import json
import pandas as pd



tasks = ["qa1", "qa2"]
#split_names = ["2k", "4k", "8k", "16k", "32k", "64k"]
split_names = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k"]
dataset_name = "RMT-team/babilong"
results_folder = "./test_res"
armt_cpt_path = "../../data/pretrained_models/RMT-Llama-3.2-1B-Instruct-8x1024-mem16-lora-babilong-qa1-5_ct-v3.1/model.safetensors"
trained_model_path = "../optimized_armt/runs/test/babilong_multitask/unsloth/Llama-3.2-1B-Instruct/lr_3e-04_d64_linear_adamw_wd1e-03_8x1024_mem16_bs64_bptt--1_from_cpt_0-1_lora_ct-v3/grouped_v2/run_1"
model_name = "unsloth/Llama-3.2-1B-Instruct"
load_trained_weights = False



torch.set_default_device("cuda:0")
dtype = torch.bfloat16
torch.set_default_dtype(dtype)
torch.set_grad_enabled(False)
# load base model
source_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    attn_implementation="flash_attention_2",
                                                    torch_dtype=dtype,
                                                    device_map="cpu")
source_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=True, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    )
source_model = get_peft_model(source_model, peft_config)
# after wrap base model in original ARMT and ARMT with grouped batching, and load pretrained weigths
# the actual segment_size for this model is segment_size - mem_size, so we will use it later
segment_size = 1024
mem_size = 16
segment_alignment = "left"
attend_to_previous_input = False
device = "cpu"
max_n_segments = 32
mem_cell_args = dict(
    base_model=source_model,
    num_mem_tokens=mem_size,
    d_mem=64,
    layers_attr="model.model.layers",
    wrap_pos=False,
    correction=True,
)

cell = AssociativeMemoryCell(**mem_cell_args)
original_model = AssociativeRecurrentWrapper(cell,
                                            segment_size=segment_size-mem_size,
                                            max_n_segments=max_n_segments,
                                            segment_alignment=segment_alignment,
                                            attend_to_previous_input=attend_to_previous_input,
).to(device)

if "safetensors" in armt_cpt_path:
    from safetensors.torch import load_model
    load_model(original_model, armt_cpt_path, device="cuda:0")
else:
    cpt = torch.load(armt_cpt_path, map_location=device)
    original_model.load_state_dict(cpt, strict=True)
original_model.to("cuda")
# merge lora
merge_and_save = True
unmerge_and_save = False
if merge_and_save or unmerge_and_save:
    if merge_and_save:
        original_model.memory_cell.model.merge_and_unload()
    if unmerge_and_save:
        original_model.memory_cell.model.unload()
    original_model.memory_cell.model = original_model.memory_cell.model.base_model.model


if load_trained_weights:
    from grouped_batching.llama1b_grouping import (
        wrap_model_with_armt, get_grouped_states, 
        make_grouped_layer_from_single_layer, make_grouped_model_from_naive,
        make_grouped_sliced_layer_from_single_layer
    )
    from grouped_batching.batching import GroupedBatcher
    from grouped_batching.executor import ArmtGroupedExecutor
    from grouped_batching.fast_executor import FastGroupedArmtExecutor, GroupedLayerContext, associate_with_context, update_mem_with_context
    
    torch.autograd.set_detect_anomaly(True)
    from grouped_batching.llama1b_grouping_autograd import make_grouped_training_layer_from_single_layer, make_grouped_sliced_training_layer_from_single_layer
    dtype = torch.bfloat16
    device = "cuda"
    original_model_copy = copy.deepcopy(original_model.to(dtype))
    grouped_context = GroupedLayerContext()
    grouped_context.is_training = True
    grouped_layer = make_grouped_sliced_training_layer_from_single_layer(
        grouped_context, copy.deepcopy(original_model_copy.memory_cell.model.model.layers[0]), original_model_copy.memory_cell.model.model.layers
    )
    grouped_layer = grouped_layer.to(dtype)
    grouped_layer = grouped_layer.to(device)
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(original_model_copy, grouped_layer)
    armt_grouped_model.to(device)
    remove_lm_head = False
    if remove_lm_head:
        armt_grouped_model.memory_cell.model.lm_head = torch.nn.Identity()
    executor = FastGroupedArmtExecutor(
        armt_grouped_model,
        grouped_layer,
        grouped_context,
        16,
    )
    ### ONLY FOR FAST LATENCY VERSION
    # compile full layers
    segments_input = torch.rand((16, segment_size, 2048), device="cuda", dtype=dtype)
    i, j = 0, 16
    grouped_context.start_idx = i
    grouped_context.end_idx = j
    grouped_context.is_full = True
    
    ao = associate_with_context(grouped_layer, grouped_context, segments_input[i:j])
    grouped_layer.generate_mode = True
    armt_grouped_model.memory_cell.model.model(inputs_embeds=segments_input[i:j], use_cache=False)
    update_mem_with_context(grouped_layer, grouped_context, segments_input[i:j])
    
    grouped_context.is_training = False
    load_trained_weights = False
    
    executor.grouped_layer.load_state_dict(torch.load(trained_model_path + "/grouped_layer.pth", weights_only=True))
    executor.armt_model.load_state_dict(torch.load(trained_model_path + "/armt_model.pth", weights_only=True))
    executor.vanilla_armt_model = copy.deepcopy(original_model)
else:
    # use original model, but in optimized version
    armt_model = copy.deepcopy(original_model)
    model_config = source_model.config
    grouped_context = GroupedLayerContext()
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_sliced_layer_from_single_layer(
        grouped_context, copy.deepcopy(armt_model.memory_cell.model.model.layers[0]), *grouped_states
    )
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    executor = FastGroupedArmtExecutor(
        armt_grouped_model, 
        grouped_layer, 
        grouped_context, 
        model_config.num_hidden_layers,
        copy.deepcopy(original_model),
    )
    # compile full layers
    segments_input = torch.rand((model_config.num_hidden_layers, 512, 2048), device="cuda", dtype=dtype)
    
    i, j = 0, 16
    grouped_context.start_idx = i
    grouped_context.end_idx = j
    grouped_context.is_full = True
    
    ao = associate_with_context(grouped_layer, grouped_context, segments_input[i:j])
    grouped_layer.generate_mode = True
    armt_grouped_model.memory_cell.model.model(inputs_embeds=segments_input[i:j], use_cache=False)
    update_mem_with_context(grouped_layer, grouped_context, segments_input[i:j])



model_name = model_name
use_instruction = True
use_examples = True
use_post_prompt = True
use_chat_template = True
api_url = False

models = [executor, original_model]
model_cpts = [
    "final_fast_executor_from_orig_paper_time_fix_v2_mem_patch_armt-1b-it-v2",
    "final_orig_v2_model_paper_armt-1b-it-v2",
]

"""
model = executor
model_cpt = "DEBUG_trained_fast_executor_v2_mem_patch_armt-1b-it-v2"
model_cpt = "DEBUG_fast_executor_from_orig_paper_wo_copy_v2_mem_patch_armt-1b-it-v2"

model_cpt = "fast_executor_from_orig_paper_time_fix_v2_mem_patch_armt-1b-it-v2"

#model = original_model
#model_cpt = "DEBUG_fast_orig_from_orig_paper_wo_copy_v2_mem_patch_armt-1b-it-v2"

#model_cpt = "DEBUG_fast_executor_from_orig_paper_wo_copy_wo_head_v2_mem_patch_armt-1b-it-v2"

#model = original_model
#model_cpt = "v3_orig_v2_model_paper_armt-1b-it-v2"

#model = executor.vanilla_armt_model
#model_cpt = "orig_model_paper_armt-1b-it-v2"

model = original_model
model_cpt = "orig_v2_model_paper_armt-1b-it-v2"
"""

generate_kwargs = {
    'max_new_tokens': 20,
    'max_length': None,
    'num_beams': 1,
    'do_sample': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    #'logits_processor': [NormLogitsWrapper()],
}
template_to_use = DEFAULT_TEMPLATE
print(f'prompt template:\n{template_to_use}')
for model, model_cpt in zip(models, model_cpts):
    model.name_or_path = "custom_rmt"
    model.device = "cuda"
    
    inference_time = {el: {} for el in tasks}
    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': template_to_use,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)
    
        for split_name in tqdm(split_names, desc='lengths'):
            # load dataset
            data = datasets.load_dataset(dataset_name, split_name)
            task_data = data[task]
            inference_time[task][split_name] = 0.0
    
            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(f'{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = f'./{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)
            timefile = Path(f'{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/time_{task}_{split_name}_{prompt_name}.json')
            timefile.parent.mkdir(parents=True, exist_ok=True)
    
            df = pd.DataFrame({'target': [], 'output': [], 'question': []})
    
            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']
    
                # format input text
                input_text = get_formatted_input(context, question, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=prompt_cfg['template'])
    
                if api_url:
                    # model is running via llamacpp's serve command
                    headers = {'Content-Type': 'application/json'}
                    if generate_kwargs['temperature'] is None:
                        generate_kwargs['temperature'] = 0.0
    
                    if use_chat_template:
                        input_text = [{'role': 'user', 'content': input_text}]
                        model_inputs = tokenizer.apply_chat_template(input_text, tokenize=True,
                                                                     add_generation_prompt=True)
                    else:
                        model_inputs = tokenizer.encode(input_text, add_special_tokens=True)
    
                    request_data = {'prompt': model_inputs, 'temperature': generate_kwargs['temperature']}
                    response = requests.post(api_url, headers=headers, json=request_data).json()
                    output = response['content'].strip()
                else:
                    # generate output using local model
                    if model.name_or_path in ['THUDM/chatglm3-6b-128k', 'THUDM/LongAlign-6B-64k-base', 'THUDM/LongAlign-6B-64k']:
                        # have to add special code to run chatglm as tokenizer.chat_template tokenization is not
                        # the same as in model.chat (recommended in https://huggingface.co/THUDM/chatglm3-6b-128k)
                        with torch.no_grad():
                            output, _ = model.chat(tokenizer, input_text, history=[], **generate_kwargs)
                    else:
                        if use_chat_template:
                            input_text = [{'role': 'user', 'content': input_text}]
                            model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                         return_tensors='pt', return_dict=model.name_or_path=="custom_rmt").to(model.device)
                            if model.name_or_path != "custom_rmt":
                                model_inputs = {'input_ids': model_inputs}
                        else:
                            model_inputs = tokenizer(input_text, return_tensors='pt',
                                                     add_special_tokens=True).to(model.device)
    
                        sample_length = model_inputs['input_ids'].shape[1]
                        with torch.no_grad():
                            if "executor" in model_cpt:
                                model_inputs["seg_size"] = segment_size
                                model_inputs['input_ids'] = model_inputs['input_ids'].contiguous()
                                model_inputs['attention_mask'] = model_inputs['attention_mask'].contiguous()
                                time_start = time.time()
                                output, copy_time = executor.generate(**model_inputs, **generate_kwargs)
                                time_end = time.time()
                                inference_time[task][split_name] += time_end - time_start# - copy_time
                            else:
                                time_start = time.time()
                                output = model.generate(**model_inputs, **generate_kwargs)
                                time_end = time.time()
                                inference_time[task][split_name] += time_end - time_start
                            # we need to reset memory states between samples for activation-beacon models
                            if 'activation-beacon' in model.name_or_path and hasattr(model, 'memory'):
                                model.memory.reset()
                        if "executor" in model_cpt:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        if model.name_or_path != "custom_rmt":
                            output = output[0][sample_length:]
                        else:
                            output = output[0]
                        output = tokenizer.decode(output, skip_special_tokens=True).strip()
                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                json.dump(inference_time, open(timefile, 'w'), indent=4)
                df.to_csv(outfile, escapechar='\\')
