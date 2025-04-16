import torch
from torch import nn
from grouped_batching.linear_grouped_forward import group_gemm_jit, group_gemm_naive


gemm_fn_forward = None
def grouped_fn_forward(x, w_group):
    global gemm_fn_forward
    x_prep = x.contiguous()
    if gemm_fn_forward is None:
        gemm_fn_forward = group_gemm_jit(x_prep.unbind(0), w_group, use_efficient_allocation=False)
        
    res = gemm_fn_forward.run(x_prep.unbind(0), w_group)
    if isinstance(res, list):
        return torch.stack(res).contiguous()
    return res

gemm_fn_backward = None
def backward_fn_grouped(o_grad, w_t_group):
    global gemm_fn_backward
    o_grad_prep = o_grad.contiguous()
    if gemm_fn_backward is None:
        gemm_fn_backward = group_gemm_jit(o_grad_prep.unbind(0), w_t_group, use_efficient_allocation=False)
        
    res = gemm_fn_backward.run(o_grad_prep.unbind(0), w_t_group)
    if isinstance(res, list):
        return torch.stack(res).contiguous()
    return res

class GroupedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *w_group):
        # TODO: unfortunately, we need to save contiguous tensors here, or cutlass will fail
        # TODO: maybe we can specify layout of tensor for cutlass
        ctx.save_for_backward(x.clone(), *[w.T.contiguous() for w in w_group])
        return grouped_fn_forward(x, w_group)
    
    @staticmethod
    def backward(ctx, o_grad):
        x = ctx.saved_tensors[0]
        w_t_group = ctx.saved_tensors[1:]

        weight_grad = x.transpose(-1, -2)@o_grad
        inp_grad = backward_fn_grouped(o_grad, w_t_group)
        return inp_grad, *[weight_grad[i] for i in range(len(o_grad))]


class GroupedGemmNaiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *w_group):
        # TODO: unfortunately, we need to save contiguous tensors here, or cutlass will fail
        # TODO: maybe we can specify layout of tensor for cutlass
        ctx.save_for_backward(x.clone(), *[w.T.contiguous() for w in w_group])
        return torch.stack(group_gemm_naive(x, w_group))
    
    @staticmethod
    def backward(ctx, o_grad):
        x = ctx.saved_tensors[0]
        w_t_group = ctx.saved_tensors[1:]

        weight_grad = x.transpose(-1, -2)@o_grad
        inp_grad = torch.stack(group_gemm_naive(o_grad, w_t_group))
        return inp_grad, *[weight_grad[i] for i in range(len(o_grad))]
    
grouped_gemm_autograd_fn = GroupedGemmFunction.apply
grouped_gemm_naive_autograd_fn = GroupedGemmNaiveFunction.apply


class GroupedLinear(nn.Module):
    def __init__(self, n_layers, in_features, out_features, dtype=torch.bfloat16, bias=False, device='cuda', use_naive_implementation=False):
        super().__init__()
        assert device == 'cuda'
        self.wg = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, out_features, dtype=dtype, device=device))
            for _ in range(n_layers)
        ])
        self.bias = nn.Parameter(torch.zeros((n_layers, 1, out_features), dtype=dtype, device=device)) if bias else None
        self.use_naive_implementation = use_naive_implementation
    @classmethod
    def from_torch_layers(cls, torch_layers: list[nn.Linear], device='cuda', use_naive_implementation=False):
        layer = cls(
            len(torch_layers), 
            torch_layers[0].in_features, 
            torch_layers[0].out_features, 
            dtype=torch_layers[0].weight.dtype, 
            bias=torch_layers[0].bias is not None, 
            device=device,
            use_naive_implementation=use_naive_implementation
        )
        for i, l in enumerate(torch_layers):
            layer.wg[i].data.copy_(l.weight.data.T)
            
            if l.bias is not None:
                # manually expand dimensions for correct forward broadcast
                layer.bias.data.copy_(l.bias.data[None, None, :])
        return layer


    def forward(self, x):
        if self.use_naive_implementation:
            out = grouped_gemm_naive_autograd_fn(x, *self.wg)
        else:
            out = grouped_gemm_autograd_fn(x, *self.wg)
        if self.bias is not None:
            out += self.bias
        return out
