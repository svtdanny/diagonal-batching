import torch
from typing import Iterator, Dict, List, Any, Union, Optional, Callable
import re
import transformers


def extract_params_from_module(module: torch.nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
    """
    Recursively extract all parameters from a module with their full path names.
    
    Args:
        module: The module to extract parameters from
        prefix: Prefix for parameter names (used in recursion)
        
    Returns:
        Dictionary mapping parameter paths to tensor values
    """
    params = {}
    
    for name, param in module.named_parameters(recurse=False):
        param_name = f"{prefix}.{name}" if prefix else name
        params[param_name] = param
    
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_params = extract_params_from_module(child, child_prefix)
        params.update(child_params)
    
    return params


def get_universal_grouped_states(
    layers_iterator: Iterator[torch.nn.Module],
    norm_pattern: str = r'(norm|layernorm|ln_\d+)\.weight$',
    special_attrs: List[str] = None
) -> Dict[str, List[torch.Tensor]]:
    """
    Universal grouping function that extracts parameters from model layers.
    
    Args:
        layers_iterator: Iterator over model layers
        norm_pattern: Regex pattern to identify normalization layer weights
        special_attrs: List of special attributes to extract that aren't registered parameters
        
    Returns:
        Dictionary where keys are parameter names and values are lists of
        tensors grouped across layers
    """
    # Convert iterator to list to avoid exhausting it
    layers = list(layers_iterator)
    if not layers:
        return {}
    
    # Compile the norm pattern regex
    norm_regex = re.compile(norm_pattern)
    
    # Extract all parameters from the first layer to identify parameter names
    first_layer_params = extract_params_from_module(layers[0])
    
    # Initialize dictionary to store grouped parameters
    grouped_params: Dict[str, List[torch.Tensor]] = {
        param_name: [] for param_name in first_layer_params.keys()
    }
    
    # Special attributes to extract that aren't registered parameters
    if special_attrs is None:
        special_attrs = ['W_mem', 'z']
        
    for attr in special_attrs:
        if hasattr(layers[0], attr):
            grouped_params[attr] = []
    
    # For each layer, extract and group parameters
    for layer in layers:
        layer_params = extract_params_from_module(layer)
        
        # Process registered parameters
        for param_name, param_tensor in layer_params.items():
            # Check if this parameter exists in our dictionary
            if param_name in grouped_params:
                # Process the tensor based on common conventions
                if param_name.endswith('weight') and not norm_regex.search(param_name):
                    # For most weight matrices, transpose and make contiguous
                    # This matches the convention in llama1b_grouping.py
                    grouped_params[param_name].append(param_tensor.data.T.contiguous())
                else:
                    # For biases, norms, etc., just make contiguous
                    grouped_params[param_name].append(param_tensor.data.contiguous())
            else:
                # Handle the case where a parameter exists in later layers but not the first
                grouped_params[param_name] = [param_tensor.data.contiguous()]
        
        # Process special attributes
        for attr in special_attrs:
            if attr in grouped_params and hasattr(layer, attr):
                # Get the special attribute and make it contiguous
                attr_tensor = getattr(layer, attr)
                if isinstance(attr_tensor, torch.Tensor):
                    grouped_params[attr].append(attr_tensor.data.contiguous())
    
    # For parameters that need to be stacked rather than kept as separate tensors
    # (similar to norm parameters in the original implementation)
    for param_name in list(grouped_params.keys()):
        if norm_regex.search(param_name):
            # Stack normalization weights
            grouped_params[param_name] = torch.stack(grouped_params[param_name]).contiguous()
    
    return grouped_params


# def get_sliced_layer_norm_forward(self, context):
#     """
#     Creates a forward function for LayerNorm that supports grouped execution.
    
#     This is a replacement for the standard PyTorch LayerNorm forward function
#     that allows for sliced execution based on the context.
    
#     Args:
#         self: The LayerNorm module
#         context: The context object containing information about the current group
        
#     Returns:
#         A forward function that implements sliced LayerNorm
#     """
#     def forward(hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
        
#         # LayerNorm uses both mean and variance, unlike RMSNorm which uses only variance
#         mean = hidden_states.mean(-1, keepdim=True)
#         var = hidden_states.pow(2).mean(-1, keepdim=True) - mean.pow(2)
        
#         # Add a small epsilon for numerical stability
#         hidden_states = (hidden_states - mean) * torch.rsqrt(var + self.eps)
        
#         if context.is_full:
#             # Use the full parameter set
#             if self.bias is not None:
#                 return (self.weight * hidden_states + self.bias).to(input_dtype)
#             else:
#                 return (self.weight * hidden_states).to(input_dtype)
#         else:
#             # Use sliced parameters based on context
#             if self.bias is not None:
#                 return (self.weight[context.start_idx:context.end_idx] * hidden_states + 
#                         self.bias[context.start_idx:context.end_idx]).to(input_dtype)
#             else:
#                 return (self.weight[context.start_idx:context.end_idx] * hidden_states).to(input_dtype)
    
#     return forward

# import torch.nn.functional as F
# def get_sliced_layer_norm_forward(self, context):
#     def forward(group_ln_inp):
#         if context.is_full:
#             grouped_out = torch.vmap(
#                 lambda x, w, b: F.layer_norm(x, self.normalized_shape, w.squeeze(0), b.squeeze(0), self.eps)
#             )(group_ln_inp, self.weight, self.bias)
#         else:
#             grouped_out = torch.vmap(
#                 lambda x, w, b: F.layer_norm(x, self.normalized_shape, w.squeeze(0), b.squeeze(0), self.eps)
#             )(group_ln_inp, self.weight[context.start_idx:context.end_idx], self.bias[context.start_idx:context.end_idx])
            
#         return grouped_out
#     return forward


import torch.nn.functional as F
def get_sliced_layer_norm_forward(self, context):
    def forward(group_ln_inp):
        if context.is_full:
            out_loop = torch.stack([
                F.layer_norm(group_ln_inp[i], self.normalized_shape,
                            self.weight[i, 0], self.bias[i, 0], self.eps)
                for i in range(self.weight.shape[0])
            ])
        else:
            w_use = self.weight[context.start_idx:context.end_idx]
            b_use = self.bias[context.start_idx:context.end_idx]
            
            out_loop = torch.stack([
                F.layer_norm(group_ln_inp[i], self.normalized_shape,
                            w_use[i, 0], b_use[i, 0], self.eps)
                for i in range(group_ln_inp.shape[0])
            ])
        return out_loop
    return forward


def get_module_by_path(root_module: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    """
    Navigate to a module by its path.
    
    Args:
        root_module: The root module to start navigation from
        path: The dot-separated path to the target module
        
    Returns:
        The target module or None if not found
    """
    if not path:
        return root_module
        
    components = path.split('.')
    current = root_module
    
    for component in components:
        if hasattr(current, component):
            current = getattr(current, component)
        else:
            return None
    
    return current


def make_universal_grouped_layer(
    context,
    grouped_layer,
    grouped_params: Dict[str, List[torch.Tensor]],
    device='cuda',
    grouped_fn=None,
    naive_grouped_fn=None,
    sliced_norm_fn=None,
    norm_pattern: str = r'(norm|layernorm|ln_\d+)\.weight$',
    special_attrs: List[str] = None,
    use_layer_norm: bool = True
):
    """
    Universal version of grouped layer creation that works with parameters dictionary
    from get_universal_grouped_states.
    
    Args:
        context: Context for sliced forward functions
        grouped_layer: The layer to modify with grouped parameters
        grouped_params: Dictionary of grouped parameters from get_universal_grouped_states
        device: Device to place tensors on
        grouped_fn: Function for applying grouped GEMM operation (if None, imported from module)
        naive_grouped_fn: Function for applying naive grouped operation (if None, imported from module) 
        sliced_norm_fn: Function for applying sliced norm (if None, imported from module)
        norm_pattern: Regex pattern to identify normalization layer weights
        special_attrs: List of special attributes that are not registered parameters
        use_layer_norm: Whether to use LayerNorm (True) or RMSNorm (False) implementation
        
    Returns:
        Modified grouped_layer with grouped execution
    """
    # Import functions if not provided
    if grouped_fn is None or naive_grouped_fn is None:
        from grouped_batching.linear_grouped_sliced_forward import (
            get_grouped_gemm_sliced_forward,
            get_naive_grouped_sliced_forward
        )
        
        if grouped_fn is None:
            grouped_fn = get_grouped_gemm_sliced_forward
            # grouped_fn = get_naive_grouped_sliced_forward
        if naive_grouped_fn is None:
            naive_grouped_fn = get_naive_grouped_sliced_forward
    
    # Select the appropriate norm function based on use_layer_norm flag
    if sliced_norm_fn is None:
        if use_layer_norm:
            sliced_norm_fn = get_sliced_layer_norm_forward
        else:
            from grouped_batching.linear_grouped_sliced_forward import get_sliced_rms_norm_forward
            sliced_norm_fn = get_sliced_rms_norm_forward
    
    # Compile the norm pattern regex
    norm_regex = re.compile(norm_pattern)
    
    # Default special attributes if not provided
    if special_attrs is None:
        special_attrs = ['W_mem', 'z']
    
    # Process special attributes (like W_mem, z) which are not registered parameters
    for attr_name in special_attrs:
        if attr_name in grouped_params:
            attr_values = grouped_params[attr_name]
            if isinstance(attr_values, list) and attr_values:
                try:
                    # Handle direct attributes on the grouped layer
                    if hasattr(grouped_layer, attr_name):
                        # Set the data attribute directly
                        attr_object = getattr(grouped_layer, attr_name)
                        if hasattr(attr_object, 'data'):
                            attr_object.data = torch.concat(attr_values, dim=0).to(device)
                        else:
                            setattr(grouped_layer, attr_name, torch.concat(attr_values, dim=0).to(device))
                    # Try to handle attributes with dots in their name
                    else:
                        attr_path_parts = attr_name.split('.')
                        if len(attr_path_parts) > 1:
                            parent_path = '.'.join(attr_path_parts[:-1])
                            last_attr = attr_path_parts[-1]
                            parent_module = get_module_by_path(grouped_layer, parent_path)
                            if parent_module is not None and hasattr(parent_module, last_attr):
                                attr_object = getattr(parent_module, last_attr)
                                if hasattr(attr_object, 'data'):
                                    attr_object.data = torch.concat(attr_values, dim=0).to(device)
                                else:
                                    setattr(parent_module, last_attr, torch.concat(attr_values, dim=0).to(device))
                except (AttributeError, RuntimeError) as e:
                    print(f"Error setting special attribute {attr_name}: {e}")
                    # Skip attributes that can't be set
                    pass
    
    # Process weight and bias parameters
    weight_params = {}
    bias_params = {}
    
    # Group weights and their corresponding biases
    for param_name, param_values in grouped_params.items():
        # Skip norm weights and special attributes
        if isinstance(param_values, torch.Tensor) and norm_regex.search(param_name):
            continue
            
        if param_name in special_attrs:
            continue
            
        # Check if this is a weight parameter
        if param_name.endswith('.weight'):
            module_path = param_name[:-7]  # Remove '.weight'
            weight_params[module_path] = param_values
        
        # Check if this is a bias parameter
        if param_name.endswith('.bias'):
            module_path = param_name[:-5]  # Remove '.bias'
            bias_params[module_path] = param_values
    
    # Apply weights and biases to linear modules
    for module_path, weight_values in weight_params.items():
        try:
            # Navigate to the module
            target_module = get_module_by_path(grouped_layer, module_path)
            if target_module is None:
                continue
            
            
            bias_tensor = None
            if module_path in bias_params:
                bias_values = bias_params[module_path]
                # Apply the weight with bias
                bias_tensor = torch.stack(bias_values)[..., None, :].to(device) if isinstance(bias_values, list) else bias_values[..., None, :].to(device)
            
            # Handle case where the module has bias
            # if module_path in bias_params:
            if any(d < 32 for d in weight_values[0].shape):
                print(f"SUBSTITUTE naive {module_path=}: {len(weight_values)=}"
                      f" {len(bias_tensor) if bias_tensor is not None else None} ")
                target_module.forward = naive_grouped_fn(context, weight_values, bias_tensor)
            else:                
                # Apply the weight without bias
                print(f"SUBSTITUTE efficient {module_path=}: {len(weight_values)=}"
                      f" {len(bias_tensor) if bias_tensor is not None else None} ")
                target_module.forward = grouped_fn(context, weight_values, bias_tensor)
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error setting module {module_path}: {e}")
            # Skip modules that don't match expectations
            pass
            raise
    
    # Process norm layers - stack them if not already stacked
    for param_name, param_values in grouped_params.items():
        if norm_regex.search(param_name):
            # Extract the component path for the norm layer
            norm_path = param_name.replace('.weight', '')
            
            try:
                # Navigate to the norm module
                norm_module = get_module_by_path(grouped_layer, norm_path)
                if norm_module is not None and hasattr(norm_module, 'weight'):
                    # For tensor param_values (already stacked), use as is
                    if isinstance(param_values, torch.Tensor):
                        norm_weight = param_values
                    # For list param_values, stack them
                    elif isinstance(param_values, list) and param_values:
                        norm_weight = torch.stack(param_values).contiguous()
                    else:
                        continue
                    
                    # Apply the norm weights
                    norm_module.weight.data = norm_weight[:, None, :]
                    
                    norm_bias = None
                    
                    # Check if the norm module has bias and apply it
                    if hasattr(norm_module, 'bias') and norm_module.bias is not None:
                        # Get the corresponding bias parameter
                        bias_param_name = norm_path + '.bias'
                        if bias_param_name in grouped_params:
                            bias_values = grouped_params[bias_param_name]
                            
                            # Process bias values similar to weights
                            if isinstance(bias_values, torch.Tensor):
                                norm_bias = bias_values
                            elif isinstance(bias_values, list) and bias_values:
                                norm_bias = torch.stack(bias_values).contiguous()
                            else:
                                norm_bias = None
                                
                            # Apply the norm bias if available
                            if norm_bias is not None:
                                norm_module.bias.data = norm_bias[:, None, :]
                    
                    # Apply the forward function
                    print(f"SUBSTITUTE {norm_path=}: {norm_weight.shape if norm_weight is not None else None} {norm_bias.shape if norm_bias is not None else None} ")
                    norm_module.forward = sliced_norm_fn(norm_module, context)
            except (AttributeError, RuntimeError) as e:
                print(f"Error setting norm module {norm_path}: {e}")
                # Skip modules that don't match expectations
                raise
                pass
    
    # Mark as grouped execution
    grouped_layer._grouped_execution = True
    grouped_layer._skip_associating = True
    
    return grouped_layer 


def set_module_by_path(root_module: torch.nn.Module, path: str, value: Any) -> bool:
    """
    Set a module or attribute at a specified path.
    
    Args:
        root_module: The root module to start navigation from
        path: The dot-separated path to the target attribute
        value: The value to set
        
    Returns:
        True if successful, False otherwise
    """
    if not path:
        return False
    
    components = path.split('.')
    current = root_module
    
    # Navigate to the parent module
    for component in components[:-1]:
        if hasattr(current, component):
            current = getattr(current, component)
        else:
            return False
    
    # Set the attribute on the parent module
    try:
        setattr(current, components[-1], value)
        return True
    except (AttributeError, TypeError):
        return False

def make_universal_grouped_model(
    model: torch.nn.Module,
    grouped_layer: torch.nn.Module,
    layers_path: str = "memory_cell.model.model.layers",
    norm_path: Optional[str] = "memory_cell.model.model.norm",
    out_norm_attr: str = "out_norm"
) -> tuple:
    """
    Universal version of make_grouped_model_from_naive that works with any model structure.
    
    Args:
        model: The model to modify
        grouped_layer: The grouped layer to use
        layers_path: Path to the layers attribute in the model
        norm_path: Path to the final normalization layer (if any)
        out_norm_attr: Name of the attribute to store the original norm
        
    Returns:
        Tuple of (modified model, original layers)
    """
    # Get the layers module
    layers_module = get_module_by_path(model, layers_path)
    if layers_module is None:
        raise ValueError(f"Could not find layers at path: {layers_path}")
    
    # Save the original layers
    source_model_layers = layers_module

    # Replace the layers with the single grouped layer
    if not set_module_by_path(model, layers_path, torch.nn.ModuleList([grouped_layer])):
        raise ValueError(f"Could not set grouped layer at path: {layers_path}")
    
    return model, source_model_layers
