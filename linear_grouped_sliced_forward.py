import torch

from grouped_batching.linear_grouped_forward import group_cutlass_naive, group_gemm_naive, get_group_gemm
import cutlass

def get_naive_grouped_sliced_forward(context, Ws, bias=None, is_list = False):
    def forward(Xs):
        # print("matmul shape input: ", Xs.shape)
        
        if context.is_full:
            res_list = group_gemm_naive(Xs if is_list else Xs.unbind(0), Ws)
            res = torch.stack(res_list)
            if bias is not None:
                res += bias
        else:
            res_list = group_gemm_naive(Xs if is_list else Xs.unbind(0), Ws[context.start_idx:context.end_idx])
            res = torch.stack(res_list)
            if bias is not None:
                res += bias[context.start_idx:context.end_idx]
                
        # print("matmul shape out: ", res.shape)
        return res
    return forward

def get_naive_grouped_sliced_forward(context, Ws, bias=None, is_list = False):
    def forward(Xs):
        # print("matmul shape input: ", Xs.shape)
        
        if context.is_full:
            res_list = group_gemm_naive(Xs if is_list else Xs.unbind(0), Ws)
            res = torch.stack(res_list)
            if bias is not None:
                res += bias
        else:
            res_list = group_gemm_naive(Xs if is_list else Xs.unbind(0), Ws[context.start_idx:context.end_idx])
            res = torch.stack(res_list)
            if bias is not None:
                res += bias[context.start_idx:context.end_idx]
                
        # print("matmul shape out: ", res.shape)
        return res
    return forward

def get_grouped_gemm_sliced_forward(context, Ws, bias=None):
    gemm_fn = get_group_gemm()
    
    def forward(Xs):
        nonlocal gemm_fn
        # print(f"Xs: {Xs.shape}, Ws[0]: {Ws[0].shape}")
        Xs = Xs.contiguous()
        if context.is_full:
            res_list = gemm_fn(Xs.unbind(0), Ws)
        else:
            res_list = gemm_fn(Xs.unbind(0), Ws[context.start_idx:context.end_idx])
        # res_list = group_gemm_naive(Xs.unbind(0), Ws)
        if isinstance(res_list, list):
            res = torch.stack(res_list).contiguous()
        else:
            res = res_list  
        # print(f"Outs: {res.shape}")
        if bias is not None:
            if context.is_full:
                res += bias
            else:
                res += bias[context.start_idx:context.end_idx]
        return res
    return forward

def get_sliced_rms_norm_forward(self, context):
    def forward(hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if context.is_full:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return self.weight[context.start_idx:context.end_idx] * hidden_states.to(input_dtype)

    return forward
