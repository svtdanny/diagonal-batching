import sys
sys.path.append("/home/jovyan/sivtsov/associative-recurrent-memory-transformer")

from modeling_amt.language_modeling import AssociativeMemoryCell, AssociativeRecurrentWrapper
from transformers import AutoModelForCausalLM
import transformers

import torch

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

def group_gemm_naive(As, Bs):
    return [A @ B for A, B in zip(As, Bs)]

def get_naive_grouped_forward(Ws, bias=None):
    def forward(Xs):
        res_list = group_gemm_naive(Xs.unbind(0), Ws)
        res = torch.stack(res_list)
        # print(f"Outs: {res.shape}")
        if bias is not None:
            res += bias
        return res
    return forward

import cutlass 
import cutlass_emit_pytorch_mocked
USE_EFFICIENT_ALLOCATION = False

def group_gemm_jit(As, Bs):
    dtype = As[0].dtype
    print(f"GROUPED GEMM dtype: {dtype}")
    plan = cutlass.op.GroupedGemm(element=dtype, element_accumulator=torch.float32, layout=cutlass.LayoutType.RowMajor)

    Cs = [torch.zeros(a.shape[:-1] + (b.shape[-1],), dtype=a.dtype, device=a.device) for a,b in zip(As, Bs)]
    Ds = [torch.zeros_like(el) for el in Cs]
    
    plan.run(As, Bs, Cs, Ds, print_module=True)
    op = plan.construct()

    if USE_EFFICIENT_ALLOCATION:
        print("USE_EFFICIENT_ALLOCATION")
        import cutlass_emit_pytorch_mocked
        grouped_gemm = cutlass_emit_pytorch_mocked.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=True)
    else:
        grouped_gemm = cutlass.emit.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=True)

    return grouped_gemm

def get_group_gemm():
    gemm_fn = None
    
    def group_gemm(As, Bs):
        nonlocal gemm_fn
        if gemm_fn is None:
            print(f"jit compile As: {[a.shape for a in As]} Bs: {[b.shape for b in Bs]}")
            gemm_fn = group_gemm_jit(As, Bs)

        return gemm_fn.run(As, Bs)

    return group_gemm

def get_grouped_gemm_forward(Ws, bias=None):
    gemm_fn = get_group_gemm()
    
    def forward(Xs):
        nonlocal gemm_fn
        # print(f"Xs: {Xs.shape}, Ws[0]: {Ws[0].shape}")
        Xs = Xs.contiguous()
        res_list = gemm_fn(Xs.unbind(0), Ws)
        # res_list = group_gemm_naive(Xs.unbind(0), Ws)
        if isinstance(res_list, list):
            res = torch.stack(res_list).contiguous()
        else:
            res = res_list  
        # print(f"Outs: {res.shape}")
        if bias is not None:
            res += bias
        return res
    return forward

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


class GroupedBatcher:
    def __init__(self, armt_model, n_layers, seg_size, hid_dim, pos_embed_dim):
        self.armt_model = armt_model
        self.segment_storage = []
        self.n_layers = n_layers
        self.seg_size = seg_size
        self.hid_dim = hid_dim
        self.pos_embed_dim = pos_embed_dim
        self.cur_cont = 0

        self.out_storage = {}
        
    def init_batch(self, dtype=torch.bfloat16, device="cpu"):
        batch = torch.zeros((self.n_layers, self.seg_size, self.hid_dim), dtype=dtype, device=device)
        # batch = torch.zeros((self.n_layers, self.seg_size), dtype=dtype, device=device)
        segments_info = torch.full((self.n_layers,), -1, dtype=torch.int32, device=device)
        # position_ids = torch.full((self.n_layers, self.seg_size), 0, dtype=torch.int32, device=device)
        # positional_embeddings = (torch.zeros((self.n_layers, self.seg_size, self.pos_embed_dim), dtype=dtype, device=device),
        #                         torch.zeros((self.n_layers, self.seg_size, self.pos_embed_dim), dtype=dtype, device=device))
        
        position_ids, positional_embeddings = None, None
        
        return batch, segments_info, position_ids, positional_embeddings
        
    def push(self, batch):
        self.segment_storage.append(
            [
            self.cur_cont,
            batch
            ]
        )
        segm_id = self.cur_cont
        self.cur_cont += 1

        return segm_id
        
    def next(self, batch, segments_info, position_ids, positional_embeddings):     
        batch = torch.roll(batch, 1, 0)
        prev_segments_info = segments_info
        segments_info = torch.roll(segments_info, 1, 0)
        # position_ids = torch.roll(position_ids, 1, 0)
        # positional_embeddings = (torch.roll(positional_embeddings[0], 1, 0), torch.roll(positional_embeddings[1], 1, 0))
        
        if not self.segment_storage:
            batch[0] = 0
            segments_info[0] = -1
            # position_ids[0] = 0
            # positional_embeddings[0][0] = 0
            # positional_embeddings[1][0] = 0
            need_to_zero_mem = prev_segments_info != segments_info
            return batch, segments_info, position_ids, positional_embeddings, need_to_zero_mem
        
        segments_info[0] = self.segment_storage[0][0]
        need_to_zero_mem = prev_segments_info != segments_info
        
        processed_input = self.armt_model.memory_cell.process_input(**self.segment_storage[0][1][0])
        batch[0] = processed_input["inputs_embeds"]

        self.segment_storage[0][1] = self.segment_storage[0][1][1:]
        
        if len(self.segment_storage[0][1]) == 0:
            self.segment_storage = self.segment_storage[1:]
        
        return batch, segments_info, position_ids, positional_embeddings, need_to_zero_mem
    
    def push_out(self, batch_out, segments_info):
        # print("push out segments_info", segments_info)        
        last_segm_info = segments_info[-1].item()
        if last_segm_info != -1:
            batch_out = self.armt_model.memory_cell.process_output(
                    batch_out
                    , None, None
            )
            
            if last_segm_info not in self.out_storage:
                self.out_storage[last_segm_info] = []

            # self.out_storage[last_segm_info].append(batch_out[-1:])
            self.out_storage[last_segm_info].append(batch_out)
            
            
    def get_context_output(self, segm_id):
        segm_out = self.out_storage[segm_id]
        del self.out_storage[segm_id]
        return segm_out
    

import torch.nn as nn
class ArmtGroupedExecutor(nn.Module):
    def __init__(self, armt_model, grouped_model_layer, batcher):
        super().__init__()
        self.armt_model = armt_model
        self.grouped_model_layer = grouped_model_layer
        self.batcher = batcher
        
    def forward(self, input_ids):
        segmented_input = self.armt_model.segment(
            input_ids=input_ids,
        )

        # processed_segments = [armt_model.memory_cell.process_input(**s_i) for s_i in segmented_input]
        processed_segments = segmented_input
        out_seg_id = self.batcher.push(processed_segments)
        
        batch, segments_info, batch_position_ids, batch_position_embeddings = self.batcher.init_batch(
            # dtype=processed_segments[0]['inputs_embeds'].dtype, device=processed_segments[0]['inputs_embeds'].device
            dtype=torch.get_default_dtype(), device="cuda"
        )
        
        is_first = True

        while is_first or (segments_info != -1).any():
            assert batch is not None
            
            if is_first:
                is_first = False
            
            batch, segments_info, batch_position_ids, batch_position_embeddings, need_to_zero_mem = self.batcher.next(
                batch, segments_info, batch_position_ids, batch_position_embeddings
            )
            if (segments_info == -1).all():
                break
            
            # print("batch = ", batch)
            
            self.armt_model._first_seg_mask = need_to_zero_mem
            
            # need_to_zero_mem = need_to_zero_mem.to('cuda')

            self.grouped_model_layer.first_seg = False
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if need_to_zero_mem[i].item():
            #         # print("ZEROING MEM: ", i)
            #         self.grouped_model_layer.W_mem.data[i].fill_(0)
            #         self.grouped_model_layer.z.data[i].fill_(0)
            #         # layer.zero_mem()
            self.grouped_model_layer.W_mem.data[need_to_zero_mem].fill_(0)
            self.grouped_model_layer.z.data[need_to_zero_mem].fill_(0)
            
            assoc_batch = self.grouped_model_layer.associate(batch)
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if not need_to_zero_mem[i].item():
            #         batch[i] += assoc_batch[i]
            batch[~need_to_zero_mem] += assoc_batch[~need_to_zero_mem]
            
            # out = self.armt_model.memory_cell.model(inputs_embeds=batch)
            out = self.armt_model.memory_cell.model.model(inputs_embeds=batch, use_cache=False)
            
            
            # print("LAYER LAST: ", self.armt_model.memory_cell.model.model.layers[0]._last_input)
            # print("LAYER LAST: ", out[0])
            if segments_info[-1].item() != -1:
                out_normed = self.armt_model.out_norm(out[0][-1:])
                lmhead_out = self.armt_model.memory_cell.model.lm_head(out_normed)
                # print("LMHEAD: ",lmhead_out)
                out_logits = transformers.modeling_outputs.CausalLMOutputWithPast(
                    logits=lmhead_out,
                )
            else:
                out_logits = None
            # print("CMP: ",batch.shape, out[0].shape, batch.dtype, out[0].dtype, batch, out[0])
            batch = out[0]
            # self.batcher.push_out(out.logits[-1:], segments_info)
            self.batcher.push_out(out_logits, segments_info)
            
        segm_out_logits = self.batcher.get_context_output(out_seg_id)
        # segm_outs = [transformers.modeling_outputs.CausalLMOutputWithPast(
        #     logits=sol
        # ) for sol in segm_out_logits]
        # segm_outs = segm_out_logits
        
        # out = []
        # for sol in segm_outs:
        #     # out.append(self.armt_model.memory_cell.process_output(sol, None, None))
        #     out.append(sol)
        # out = self.armt_model.process_outputs(out)
        
        out = transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=torch.cat([sol.logits for sol in segm_out_logits], dim=1),
        )
            
        return out


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

    
    with torch.no_grad():
        o0 = executor.forward(input_ids)

    torch.cuda.synchronize()
    
    print(o0)

from multiprocessing import Barrier, Lock, Process
import time

def pipeline_thread_worker(rank, barrier):
    print(f"Sleep {rank}")
    time.sleep(5)
    print(f"Wake up {rank}")
    barrier.wait()
    print(f"Done {rank}")
    
class ArmtGroupedThreadExecutor(nn.Module):
    def __init__(self, armt_model, grouped_model_layer, batcher):
        super().__init__()
        self.armt_model = armt_model
        self.grouped_model_layer = grouped_model_layer
        self.batcher = batcher
        
    def forward(self, input_ids, rank, lock):
        segmented_input = self.armt_model.segment(
            input_ids=input_ids,
        )

        # processed_segments = [armt_model.memory_cell.process_input(**s_i) for s_i in segmented_input]
        processed_segments = segmented_input
        out_seg_id = self.batcher.push(processed_segments)
        
        batch, segments_info, batch_position_ids, batch_position_embeddings = self.batcher.init_batch(
            # dtype=processed_segments[0]['inputs_embeds'].dtype, device=processed_segments[0]['inputs_embeds'].device
            dtype=torch.get_default_dtype(), device="cuda"
        )
        
        is_first = True
        
        # print(f"Waiting: {rank}")
        # barrier.wait()
        # print(f"Start compute: {rank}")
        
        while is_first or (segments_info != -1).any():
            assert batch is not None
            
            if is_first:
                is_first = False
            
            batch, segments_info, batch_position_ids, batch_position_embeddings, need_to_zero_mem = self.batcher.next(
                batch, segments_info, batch_position_ids, batch_position_embeddings
            )
            if (segments_info == -1).all():
                break
            
            # print("batch = ", batch)
            
            self.armt_model._first_seg_mask = need_to_zero_mem
            
            # need_to_zero_mem = need_to_zero_mem.to('cuda')

            self.grouped_model_layer.first_seg = False
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if need_to_zero_mem[i].item():
            #         # print("ZEROING MEM: ", i)
            #         self.grouped_model_layer.W_mem.data[i].fill_(0)
            #         self.grouped_model_layer.z.data[i].fill_(0)
            #         # layer.zero_mem()
            self.grouped_model_layer.W_mem.data[need_to_zero_mem].fill_(0)
            self.grouped_model_layer.z.data[need_to_zero_mem].fill_(0)
            
            assoc_batch = self.grouped_model_layer.associate(batch)
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if not need_to_zero_mem[i].item():
            #         batch[i] += assoc_batch[i]
            batch[~need_to_zero_mem] += assoc_batch[~need_to_zero_mem]
            
            # out = self.armt_model.memory_cell.model(inputs_embeds=batch)
            lock.acquire()
            out = self.armt_model.memory_cell.model.model(inputs_embeds=batch, use_cache=False)
            lock.release()
            
            # print("LAYER LAST: ", self.armt_model.memory_cell.model.model.layers[0]._last_input)
            # print("LAYER LAST: ", out[0])
            if segments_info[-1].item() != -1:
                out_normed = self.armt_model.out_norm(out[0][-1:])
                lmhead_out = self.armt_model.memory_cell.model.lm_head(out_normed)
                # print("LMHEAD: ",lmhead_out)
                out_logits = transformers.modeling_outputs.CausalLMOutputWithPast(
                    logits=lmhead_out,
                )
            else:
                out_logits = None
            # print("CMP: ",batch.shape, out[0].shape, batch.dtype, out[0].dtype, batch, out[0])
            batch = out[0]
            # self.batcher.push_out(out.logits[-1:], segments_info)
            self.batcher.push_out(out_logits, segments_info)
            
        segm_out_logits = self.batcher.get_context_output(out_seg_id)
        # segm_outs = [transformers.modeling_outputs.CausalLMOutputWithPast(
        #     logits=sol
        # ) for sol in segm_out_logits]
        # segm_outs = segm_out_logits
        
        # out = []
        # for sol in segm_outs:
        #     # out.append(self.armt_model.memory_cell.process_output(sol, None, None))
        #     out.append(sol)
        # out = self.armt_model.process_outputs(out)
        
        # out = transformers.modeling_outputs.CausalLMOutputWithPast(
        #     logits=torch.cat([sol.logits for sol in segm_out_logits], dim=1),
        # )
            
        # return out
        return segm_out_logits

def threaded_executor_forward_main(rank, barrier, lock):
    # torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_grad_enabled(False)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    
    armt_model = get_llama1b_model(dtype)
    armt_model.eval()
    
    seg_size = 1024
    num_mem_tokens = 256
    d_mem = 64
    
    armt_model = wrap_model_with_armt(armt_model, segment_size=seg_size, num_mem_tokens=num_mem_tokens, d_mem=d_mem)
    armt_model.to("cuda")
    
    grouped_states = get_grouped_states(armt_model)
    grouped_layer = make_grouped_layer_from_single_layer(
        armt_model.memory_cell.model.model.layers[0], *grouped_states)
    armt_grouped_model, source_model_layers = make_grouped_model_from_naive(armt_model, grouped_layer)
    
    batcher = GroupedBatcher(armt_grouped_model, n_layers=16, seg_size=seg_size+num_mem_tokens, hid_dim=2048, pos_embed_dim=2048)
    executor = ArmtGroupedThreadExecutor(armt_grouped_model, grouped_layer, batcher)

    seq_len = 512*256
    print(f"SEQ LEN: {seq_len}")
    input_ids=torch.randint(0, 10000, (1, seq_len), dtype=torch.long, device="cuda")
    
    num_warmup_steps = 5
    with torch.no_grad():
        for i in range(num_warmup_steps):
            print(f"Warmup {i}: {rank}")
            o0 = executor.forward(input_ids, rank, lock)
            del o0

    print(f"Waiting threaded_executor_forward_main: {rank}")
    barrier.wait()
    print(f"Start forward call: {rank}")
    
    start_time = time.time()
    
    with torch.no_grad():
        o0 = executor.forward(input_ids, rank, lock)

    torch.cuda.synchronize()

    end_time = time.time()
    print(f"Done forward call: {rank}, time: {end_time - start_time}")
    
    # print(o0)

if __name__ == "__main__":
    # one_simple_forward_main()

    # processes = []
    # barrier = Barrier(4)
    # for i in range(4):
    #     p = Process(target=pipeline_thread_worker, args=(i, barrier))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
        # p.join()
        
    
    processes = []
    # num_processes = 4
    num_processes = 1
    barrier = Barrier(num_processes)
    lock = Lock()
    for i in range(num_processes):
        p = Process(target=threaded_executor_forward_main, args=(i, barrier, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
