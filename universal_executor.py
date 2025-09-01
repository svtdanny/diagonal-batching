import torch
import transformers

from grouped_batching.fast_executor import (
    associate_with_context, update_mem_with_context
)

from grouped_batching.universal_grouping import (
    get_module_by_path
)


def zero_grouped_memory(self):
    """Zero out the memory of a grouped ARMT model."""
    self.W_mem.detach_()
    self.W_mem.fill_(0)
    self.z.detach_()
    self.z.fill_(0)


class UniversalGroupedExecutor(torch.nn.Module):
    """
    Universal grouped executor for ARMT models.
    
    This class provides a flexible executor that can work with any model structure
    by using configurable paths to model components.
    """
    
    def __init__(
        self, 
        model, 
        grouped_layer, 
        context, 
        n_layers, 
        model_path="memory_cell.model.model",
        out_norm_attr="out_norm",
        lm_head_path="memory_cell.model.lm_head",
        memory_path="memory_cell",
        segment_fn=None,
        process_input_fn=None,
        vanilla_model=None,
        preprocess_segment_fn = None,
        postprocess_segment_fn = None,
        grouped_compute_fn = None,
    ):
        """
        Initialize the universal grouped executor.
        
        Args:
            model: The model to execute
            grouped_layer: The grouped layer
            context: The context for grouped execution
            n_layers: Number of layers in the model
            model_path: Path to the main model module
            out_norm_attr: Attribute name where the output norm is stored
            lm_head_path: Path to the language model head
            memory_path: Path to the memory cell module
            segment_fn: Function to segment inputs (if None, uses model.segment)
            process_input_fn: Function to process inputs (if None, uses memory_cell.process_input)
            vanilla_model: Original model for generation (optional)
        """
        super().__init__()
        self.model = model
        self.grouped_layer = grouped_layer
        self.context = context
        self.n_layers = n_layers
        self.vanilla_model = vanilla_model
        
        self.preprocess_segment_fn = preprocess_segment_fn
        self.postprocess_segment_fn = postprocess_segment_fn
        self.grouped_compute_fn = grouped_compute_fn
        
        # Store paths
        self.model_path = model_path
        self.out_norm_attr = out_norm_attr
        self.lm_head_path = lm_head_path
        self.memory_path = memory_path
        
        # Get components by path
        self.base_model = get_module_by_path(model, model_path)
        if self.base_model is None:
            raise ValueError(f"Could not find base model at path: {model_path}")

        self.out_norm = get_module_by_path(model, out_norm_attr)
        if self.out_norm is None:
            print(f"Warning: Could not find out norm at path: {out_norm_attr}")
            
        self.lm_head = get_module_by_path(model, lm_head_path)
        if self.lm_head is None:
            print(f"Warning: Could not find LM head at path: {lm_head_path}")
            
        self.memory_cell = get_module_by_path(model, memory_path)
        if self.memory_cell is None:
            raise ValueError(f"Could not find memory cell at path: {memory_path}")
            
        # Segmentation and input processing functions
        self.segment_fn = segment_fn if segment_fn is not None else model.segment
        
        if process_input_fn is not None:
            self.process_input_fn = process_input_fn
        elif hasattr(self.memory_cell, 'process_input'):
            self.process_input_fn = self.memory_cell.process_input
        else:
            raise ValueError("No process_input function provided or found")
            
        # Set generation mode
        self.grouped_layer.generate_mode = True
    
    def forward(self, input_ids, skip_concat=False):
        """
        Forward pass for the grouped model.
        
        Args:
            input_ids: Input token IDs
            skip_concat: Whether to skip concatenating outputs
            
        Returns:
            Model outputs
        """
        self.context.is_full = False
        self.context.start_idx = 0
        self.context.end_idx = 0
        
        # Zero out memory
        zero_grouped_memory(self.grouped_layer)
        
        # Segment inputs
        input_segments = [iseg for iseg in self.segment_fn(input_ids=input_ids)]
        segments = [self.process_input_fn(**iseg)['inputs_embeds'][0] for iseg in input_segments]
        
        segment_outputs = []
        grouped_input = []
        
        for i in range(self.n_layers + len(segments) - 1):
            if i < len(segments):
                # Add new segment until have one 
                new_segment = segments[i]
                if self.preprocess_segment_fn is not None:
                    new_segment = self.preprocess_segment_fn(self.model, new_segment)
                grouped_input.insert(0, new_segment)
                
            grouped_input_tensor = torch.stack(grouped_input).contiguous()
            if i < self.n_layers:
                # Compute before end_idx+=1 to skip first segment association
                if i > 0 and grouped_input_tensor.shape[0] > 1:
                    grouped_input_tensor[:-1, ...] += associate_with_context(self.grouped_layer, self.context, grouped_input_tensor[:-1, ...])
                
                # Allow more weights to be computed
                self.context.end_idx += 1
                if self.context.end_idx == self.n_layers and self.context.start_idx == 0:
                    self.context.is_full = True
            else:
                grouped_input_tensor += associate_with_context(self.grouped_layer, self.context, grouped_input_tensor)
            
            
            # Process through the grouped layer
            if self.grouped_compute_fn is not None:
                grouped_output = self.grouped_compute_fn(self.model, self.grouped_layer, grouped_input_tensor)
            else:
                grouped_output = self.grouped_layer.forward(grouped_input_tensor)
            grouped_output = grouped_output[0]
            
            # Update memory with context
            update_mem_with_context(self.grouped_layer, self.context, grouped_output[:, -self.grouped_layer.num_mem_tokens:])
            
            grouped_input = list(grouped_output.unbind(0))
            if i >= self.n_layers - 1:
                segment_out_logits = grouped_input.pop(-1)
                
                
                processed_segment = segment_out_logits[:-self.grouped_layer.num_mem_tokens]
                if self.postprocess_segment_fn is not None:
                    processed_segment = self.postprocess_segment_fn(self.model, processed_segment)
                
                segment_outputs.append(processed_segment)
                
            if i >= len(segments) - 1:
                # Reduce number of weights to be computed
                self.context.start_idx += 1
                self.context.is_full = False
              
        if skip_concat:
            return segment_outputs
        
        # Concatenate outputs
        output = torch.cat(segment_outputs, dim=0)
        
        # Return as a CausalLMOutput
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=output,
        )
    
    def generate(self, input_ids, attention_mask, seg_size, **generate_kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            seg_size: Segment size
            **generate_kwargs: Additional keyword arguments for generation
            
        Returns:
            Generated output and copy time
        """
        import time
        
        # Process the vanilla model if available
        if self.vanilla_model is not None:
            vanilla_memory_cell = get_module_by_path(self.vanilla_model, self.memory_path)
            if vanilla_memory_cell is not None:
                vanilla_memory_cell.zero_mem()
        elif hasattr(self.memory_cell, 'zero_mem'):
            self.memory_cell.zero_mem()
            
        # Handle large inputs
        if input_ids.shape[-1] > seg_size:
            # Cut last part of the segment
            last_segm = input_ids.shape[-1] // (seg_size) * (seg_size)
            if last_segm == input_ids.shape[-1]:
                last_segm -= (seg_size)
            prev_ids = input_ids[..., :last_segm]
            last_ids = input_ids[..., last_segm:]
            last_attn_mask = attention_mask[..., last_segm:] if attention_mask is not None else None
            
            print(prev_ids.shape, last_ids.shape)
            
            # Process previous segments
            _ = self.forward(prev_ids)
            
            # Process last segment
            segmented = self.segment_fn(input_ids=last_ids, attention_mask=last_attn_mask)
            final_segment = segmented[-1]
            
            # Use vanilla model for generation if available
            if self.vanilla_model is not None:
                vanilla_memory_cell = get_module_by_path(self.vanilla_model, self.memory_path)
                if vanilla_memory_cell is not None:
                    # Patch memory
                    time_start = time.time()
                    vanilla_memory_cell.memory = self.memory_cell.memory
                    
                    # Copy weights
                    for idx in range(len(vanilla_memory_cell.layers)):
                        if hasattr(self.grouped_layer, 'W_mem'):
                            vanilla_memory_cell.layers[idx].W_mem = self.grouped_layer.W_mem[idx].unsqueeze(0)
                        if hasattr(self.grouped_layer, 'z'):
                            vanilla_memory_cell.layers[idx].z = self.grouped_layer.z[idx].unsqueeze(0)
                        vanilla_memory_cell.layers[idx].first_seg = False
                    
                    time_end = time.time()
                    # print(f"final_segment: {final_segment}, attention_mask: {attention_mask}, generate_kwargs: {generate_kwargs}")
                    out = vanilla_memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
                    vanilla_memory_cell.zero_mem()
                    copy_time = time_end - time_start
                    return out, copy_time
            
            # Use memory cell for generation
            if hasattr(self.memory_cell, 'generate'):
                out = self.memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
                if hasattr(self.memory_cell, 'zero_mem'):
                    self.memory_cell.zero_mem()
                return out, 0
        else:
            # Process inputs directly
            segmented = self.segment_fn(input_ids=input_ids, attention_mask=attention_mask)
            final_segment = segmented[-1]
            
            # Use vanilla model for generation
            if self.vanilla_model is not None:
                vanilla_memory_cell = get_module_by_path(self.vanilla_model, self.memory_path)
                if vanilla_memory_cell is not None:
                    out = vanilla_memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
                    vanilla_memory_cell.zero_mem()
                    return out, 0
            
            # Use memory cell for generation
            if hasattr(self.memory_cell, 'generate'):
                out = self.memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
                if hasattr(self.memory_cell, 'zero_mem'):
                    self.memory_cell.zero_mem()
                return out, 0
        
        # Fallback
        raise ValueError("Could not generate output - no suitable generation method found")
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.grouped_layer.to(device)
        if self.vanilla_model is not None:
            self.vanilla_model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        self.grouped_layer.eval()
        if self.vanilla_model is not None:
            self.vanilla_model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        self.grouped_layer.train()
        if self.vanilla_model is not None:
            self.vanilla_model.train()
        return self 
