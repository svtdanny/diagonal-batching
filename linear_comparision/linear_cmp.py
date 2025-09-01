from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import time
from datetime import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='fla-hub/rwkv7-0.4B-world')
parser.add_argument('--warmup_iters', type=int, default=2)
parser.add_argument('--iters_per_size', type=int, default=4)
parser.add_argument('--warmup_input_size', type=int, default=16384)
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--wrap_armt', action='store_true')
parser.add_argument('--armt_segment', type=int, default=1024)
parser.add_argument('--armt_mem_tokens', type=int, default=128)
parser.add_argument('--armt_d_mem', type=int, default=64)
args = parser.parse_args()

model_name = args.model_name
warmup_iters = args.warmup_iters
iters_per_size = args.iters_per_size
warmup_input_size = args.warmup_input_size
wrap_armt = args.wrap_armt
device = 'cuda'

dtype = getattr(torch, args.dtype)


input_sizes = [4096, 8192, (8192+16384)//2, 16384, (16384+32768)//2, 32768, 65536] #, 131072]

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"args: {args}")
        print(f"kwargs: {kwargs}")
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

def append_result_and_save(result_df, dtype, model_name, input_size, time, iter, is_warmup, wrap_armt):
    result_df = pd.concat([result_df, pd.DataFrame({'model': [model_name], 'input_size': [input_size], 'time': [time], 'iter': [iter], 'is_warmup': [is_warmup], 'wrap_armt': [wrap_armt]})], ignore_index=True)
    save_path = 'result_{}_{}'.format(model_name.replace('/', '__'), dtype)
    if wrap_armt:
        save_path += '_wrap_armt'
    save_path += '.csv'
    result_df.to_csv(save_path, index=False)
    
    return result_df

# Load model and tokeniser
print(f"{get_timestamp()} Loading model {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

vocab_size = int(getattr(model.config, "vocab_size", 50257))
max_pos_embeddings = getattr(model.config, "max_position_embeddings", None)

# Ensure pad token exists to avoid warnings/issues during generation
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

# Determine model's maximum usable input length for generation
# For absolute position embedding models (e.g., GPT-Neo), we must keep
# input_length + max_new_tokens <= max_position_embeddings
# if max_pos_embeddings is None or max_pos_embeddings <= 0:
    # Fallback to a conservative cap if not defined
max_pos_embeddings = 64001
max_new_tokens = 1
max_usable_input_len = max(1, max_pos_embeddings - max_new_tokens)
print(f"{get_timestamp()} Max position embeddings: {max_pos_embeddings}, using max input length {max_usable_input_len}")

if wrap_armt:
    from build_armt_for_benchmark import build_armt_for_benchmark
    armt_config = dict(
        segment_size=args.armt_segment,
        num_mem_tokens=args.armt_mem_tokens,
        d_mem=args.armt_d_mem,
    )
    _, __, model = build_armt_for_benchmark(model_name, model, armt_config, dtype, device)
else:
    model = model.to(device)

print(f"{get_timestamp()} Model loaded")

result_df = pd.DataFrame(columns=['model', 'input_size', 'time', 'iter', 'is_warmup'])

additional_kwargs = {}
if wrap_armt:
    additional_kwargs['seg_size'] = armt_config['segment_size']

print(f"{get_timestamp()} Starting warmup for model itself")
for input_size in [warmup_input_size]:
    for i in range(iters_per_size):
        # effective_input_size = min(input_size, max_usable_input_len)
        effective_input_size = input_size
        # if wrap_armt:
        #     effective_input_size += armt_config['num_mem_tokens']
        
        print(f"{get_timestamp()} Iteration {i+1} of {iters_per_size} for input size {input_size} (effective {effective_input_size})")
        input_ids = torch.randint(0, vocab_size, (1, effective_input_size), device=device, dtype=torch.long)
        # if not wrap_armt:
        attention_mask = torch.ones_like(input_ids, device=device)
        r, res_time = timeit(model.generate)(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **additional_kwargs)
        del r
        # torch.cuda.empty_cache()
        result_df = append_result_and_save(result_df, dtype, model_name, effective_input_size, res_time, i, True, wrap_armt)



for input_size in [warmup_input_size] + input_sizes:
    effective_input_size = input_size
    # if wrap_armt:
    #     effective_input_size += armt_config['num_mem_tokens']
    input_ids = torch.randint(0, vocab_size, (1, effective_input_size), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    # for i in range(warmup_iters):
    #     print(f"Warmup {i+1} of {warmup_iters} for input size {input_size}")
    #     r, time =timeit(model.generate)(inp, max_new_tokens=1)
    #     del r
    #     # torch.cuda.empty_cache()
    #     append_result_and_save(result_df, model_name, input_size, time, i, True)
        

    for i in range(iters_per_size):
        print(f"{get_timestamp()} Iteration {i+1} of {iters_per_size} for input size {input_size} (effective {effective_input_size})")
        r, res_time = timeit(model.generate)(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **additional_kwargs)
        del r
        # torch.cuda.empty_cache()
        result_df = append_result_and_save(result_df, dtype, model_name, effective_input_size, res_time, i, False, wrap_armt)
        
print(result_df)
