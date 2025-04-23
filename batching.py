import torch

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
            
            need_to_associate_mem = (prev_segments_info == segments_info) & (segments_info != -1)
            need_to_zero_mem = (prev_segments_info != segments_info) & (segments_info != -1)
            return batch, segments_info, position_ids, positional_embeddings, need_to_zero_mem, need_to_associate_mem
        
        segments_info[0] = self.segment_storage[0][0]
        need_to_zero_mem = prev_segments_info != segments_info
        need_to_associate_mem = (prev_segments_info == segments_info) & (prev_segments_info != -1)
        need_to_zero_mem = (prev_segments_info != segments_info) & (prev_segments_info != -1)
        
        processed_input = self.armt_model.memory_cell.process_input(**self.segment_storage[0][1][0])
        batch[0] = processed_input["inputs_embeds"]

        self.segment_storage[0][1] = self.segment_storage[0][1][1:]
        
        if len(self.segment_storage[0][1]) == 0:
            self.segment_storage = self.segment_storage[1:]
        
        return batch, segments_info, position_ids, positional_embeddings, need_to_zero_mem, need_to_associate_mem
    
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
