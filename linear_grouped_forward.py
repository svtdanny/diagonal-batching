import torch
import cutlass 
import grouped_batching.cutlass_emit_pytorch_mocked as cutlass_emit_pytorch_mocked

# allocate output as single tensor. Works for matrix sizes multiplication.
USE_EFFICIENT_ALLOCATION = True

def group_gemm_naive(As, Bs):
    return [A @ B for A, B in zip(As, Bs)]

def get_naive_grouped_forward(Ws, bias=None):
    def forward(Xs):
        res_list = group_gemm_naive(Xs.unbind(0), Ws)
        res = torch.stack(res_list)
        if bias is not None:
            res += bias
        return res
    return forward

def group_gemm_jit(As, Bs, use_efficient_allocation=False):
    dtype = As[0].dtype
    print(f"GROUPED GEMM dtype: {dtype}")
    plan = cutlass.op.GroupedGemm(element=dtype, element_accumulator=torch.float32, layout=cutlass.LayoutType.RowMajor)

    Cs = [torch.zeros(a.shape[:-1] + (b.shape[-1],), dtype=a.dtype, device=a.device) for a,b in zip(As, Bs)]
    Ds = [torch.zeros_like(el) for el in Cs]
    
    plan.run(As, Bs, Cs, Ds, print_module=True)
    op = plan.construct()

    if use_efficient_allocation or USE_EFFICIENT_ALLOCATION:
        print("USE_EFFICIENT_ALLOCATION")
        import grouped_batching.cutlass_emit_pytorch_mocked as cutlass_emit_pytorch_mocked
        grouped_gemm = cutlass_emit_pytorch_mocked.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=True)
    else:
        grouped_gemm = cutlass.emit.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=True)

    return grouped_gemm

gemm_fn = None
def get_group_gemm():
    def group_gemm(As, Bs):
        global gemm_fn
        # gemm_fn = group_gemm_jit(As, Bs)
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