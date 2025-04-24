
import torch.nn as nn
import torch
import transformers


class ArmtGroupedExecutor(nn.Module):
    def __init__(self, armt_model, grouped_model_layer, batcher):
        super().__init__()
        self.armt_model = armt_model
        self.grouped_model_layer = grouped_model_layer
        self.batcher = batcher
        
    def forward(self, input_ids):
        # TODO: remove this and implement zero_mem correctly
        self.armt_model.memory_cell.model.model.layers[0].W_mem.fill_(0)
        self.armt_model.memory_cell.model.model.layers[0].z.fill_(0)
        
        is_tensor_input = not isinstance(input_ids, list)
        if is_tensor_input:
            input_ids = [input_ids]
        
        segmented_input = [self.armt_model.segment(
            input_ids=ii,
        ) for ii in input_ids]

        # processed_segments = [armt_model.memory_cell.process_input(**s_i) for s_i in segmented_input]
        # processed_segments = segmented_input
        # out_seg_id = self.batcher.push(processed_segments)
        out_seg_ids = [self.batcher.push(s_i) for s_i in segmented_input]
        
        batch, segments_info, batch_position_ids, batch_position_embeddings = self.batcher.init_batch(
            # dtype=processed_segments[0]['inputs_embeds'].dtype, device=processed_segments[0]['inputs_embeds'].device
            dtype=torch.get_default_dtype(), device="cuda"
        )
        
        is_first = True

        while is_first or (segments_info != -1).any():
            assert batch is not None
            
            if is_first:
                is_first = False
            
            batch, segments_info, batch_position_ids, batch_position_embeddings, need_to_zero_mem, need_to_associate_mem = self.batcher.next(
                batch, segments_info, batch_position_ids, batch_position_embeddings
            )
            if (segments_info == -1).all():
                break
            
            # print("batch = ", batch)
            
            self.armt_model.memory_cell.model.model.layers[0]._first_seg_mask = need_to_zero_mem
            self.armt_model.memory_cell.model.model.layers[0]._need_to_update_mem = segments_info != -1
            
            # need_to_zero_mem = need_to_zero_mem.to('cuda')

            self.grouped_model_layer.first_seg = False
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if need_to_zero_mem[i].item():
            #         # print("ZEROING MEM: ", i)
            #         self.grouped_model_layer.W_mem.data[i].fill_(0)
            #         self.grouped_model_layer.z.data[i].fill_(0)
            #         # layer.zero_mem()
            self.grouped_model_layer.W_mem.data[need_to_zero_mem] = 0
            self.grouped_model_layer.z.data[need_to_zero_mem] = 0
            
            assoc_batch = self.grouped_model_layer.associate(batch)
            # for i in range(self.grouped_model_layer.W_mem.data.shape[0]):
            #     if not need_to_zero_mem[i].item():
            #         batch[i] += assoc_batch[i]
            batch[need_to_associate_mem] += assoc_batch[need_to_associate_mem]
            
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
            
        # segm_out_logits = self.batcher.get_context_output(out_seg_id)
        segm_out_logits = [self.batcher.get_context_output(out_seg_id) for out_seg_id in out_seg_ids]
        
        # segm_outs = [transformers.modeling_outputs.CausalLMOutputWithPast(
        #     logits=sol
        # ) for sol in segm_out_logits]
        # segm_outs = segm_out_logits
        
        # out = []
        # for sol in segm_outs:
        #     # out.append(self.armt_model.memory_cell.process_output(sol, None, None))
        #     out.append(sol)
        # out = self.armt_model.process_outputs(out)
        
        list_out = []
        
        for sols in segm_out_logits:
            out = transformers.modeling_outputs.CausalLMOutputWithPast(
                logits=torch.cat([sol.logits for sol in sols], dim=1),
            )
            list_out.append(out)
            
        if len(list_out) == 1 and is_tensor_input:
            return list_out[0]
        else:
            return list_out
