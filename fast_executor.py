from dataclasses import dataclass

import torch
import transformers

@dataclass
class GroupedLayerContext:
    start_idx: int = 0
    end_idx: int = 0
    is_full: bool = True
    is_training: bool = False
    
def zero_grouped_memory(self):
    self.memory_cell.model.model.layers[0].W_mem.detach_()
    self.memory_cell.model.model.layers[0].W_mem.fill_(0)
    self.memory_cell.model.model.layers[0].z.detach_()
    self.memory_cell.model.model.layers[0].z.fill_(0)

def associate_with_context(self, context, hidden_states):
    self.W_mem = self.W_mem.to(hidden_states.device)
    self.z = self.z.to(hidden_states.device)
    
    mq = self.phi(self.W_mq(hidden_states)) # (bsz, seq_len, 2d_mem * nu)

    num = torch.einsum('ijk,ikt->ijt', mq, self.W_mem[context.start_idx:context.end_idx, ...])
    denom = torch.einsum("ik,ijk->ij", self.z[context.start_idx:context.end_idx, ...], mq)[..., None] + 1e-5
    hidden_states = num / denom

    return hidden_states    

def update_mem_with_context(self, context, mem_tokens):
    self.W_mem = self.W_mem.to(mem_tokens.device)
    self.z = self.z.to(mem_tokens.device)

    mk = self.phi(self.W_mk(mem_tokens))
    new_mv = self.W_mv(mem_tokens) # (bsz, num_mem_tokens, d_model)
    
    
    
    num = torch.einsum('ijk,ikt->ijt', mk, self.W_mem[context.start_idx:context.end_idx, ...])
    denom = torch.einsum("ij,ikj->ik", self.z[context.start_idx:context.end_idx, ...], mk)[..., None] + 1e-5
    prev_mv = num / denom
    if self.correction:
        new_info_coef = 1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2 + 1e-5)[..., None]
        new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
    else:
        new_info_coef = torch.ones((context.end_idx - context.start_idx,), device=self.W_mem.data.device)
    
    if not context.is_full and context.start_idx == 0:
        # only last segment in input can be first segment to enter model
        prev_mv[-1] = 0
        new_info_coef[-1] = 1
    
    mv = new_mv - prev_mv
    mb = torch.sigmoid(self.W_mb(mem_tokens))[..., 0]
    associations =  torch.einsum('ijk,ijt,ij->ikt', mk, mv, mb) # (bsz, d_mem, d_model)

    if context.is_training:
        self.W_mem = self.W_mem.clone()
        self.z = self.z.clone()
    self.W_mem[context.start_idx:context.end_idx, ...] += associations
    self.z[context.start_idx:context.end_idx, ...] += (new_info_coef*mk).sum(dim=1)
    
    self.seg_num += 1

class FastGroupedArmtExecutor:
    def __init__(self, model, grouped_layer, context, n_layers):
        self.armt_model = model
        self.grouped_layer = grouped_layer
        self.context = context
        self.n_layers = n_layers

        self.grouped_layer.generate_mode = True

    def forward(self, input_ids, skip_concat=False): #, segments):
        self.context.is_full = False
        self.context.start_idx = 0
        self.context.end_idx = 0
        
        zero_grouped_memory(self.armt_model)
        input_segments = [iseg['input_ids'] for iseg in self.armt_model.segment(input_ids=input_ids)]
        segments = [self.armt_model.memory_cell.process_input(iseg)['inputs_embeds'][0] for iseg in input_segments]

        segment_outputs = []
        grouped_input = []
        
        for i in range(self.n_layers + len(segments) - 1):
            if i < len(segments):
                # print("insert segment shape: ", segments[i].shape)
                # add new segment until have one 
                grouped_input.insert(0, segments[i])
                
            if i < self.n_layers:
                # compute before end_idx+=1 to skip first segment association
                grouped_input_tensor = torch.stack(grouped_input)
                if i > 0 and grouped_input_tensor.shape[0] > 1:
                    # print("associate_with_context")
                    grouped_input_tensor[:-1, ...] += associate_with_context(self.grouped_layer, self.context, grouped_input_tensor[:-1, ...])
                
                # allow more weights to be computed
                self.context.end_idx += 1
                if self.context.end_idx == self.n_layers and self.context.start_idx == 0:
                    self.context.is_full = True
            else:
                grouped_input_tensor = torch.stack(grouped_input)
                grouped_input_tensor += associate_with_context(self.grouped_layer, self.context, grouped_input_tensor)
            
            # print(grouped_input_tensor.shape)
            grouped_output = self.armt_model.memory_cell.model.model(inputs_embeds=grouped_input_tensor, use_cache=False)
            grouped_output = grouped_output.last_hidden_state
            # print(type(grouped_output), grouped_output.shape)
            
            # print(f"Cur indexes: {self.context.start_idx}-{self.context.end_idx} cur i: layers={i}/{self.n_layers} segments={i}/{len(segments)}")
            update_mem_with_context(self.grouped_layer, self.context, grouped_output[:, -self.grouped_layer.num_mem_tokens:])
            
            grouped_input = list(grouped_output.unbind(0))
            if i >= self.n_layers - 1:
                segment_out_logits = grouped_input.pop(-1)
                segment_out_logits = self.armt_model.out_norm(segment_out_logits[:-self.grouped_layer.num_mem_tokens])
                segment_outputs.append(segment_out_logits)
                
            if i >= len(segments) - 1:
                # reduce number of weights to be computed
                self.context.start_idx += 1
                self.context.is_full = False
              
        if skip_concat:
            return segment_outputs
        
        # return segment_outputs  
        output = torch.cat(segment_outputs, dim=0)
        # return output
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=output,
        )
