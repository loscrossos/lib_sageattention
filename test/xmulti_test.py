import torch
import importlib
from typing import Optional

def print_header(text: str):
    print("\n" + "="*50)
    print(text)
    print("="*50)




import torch
import triton
from triton import language as tl  # Alternative import style

# Define kernel with proper scoping
@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: 'tl.constexpr',  # Note the string annotation
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_triton() -> Optional[str]:
    """Test if Triton is properly installed and functional."""
    try:
        # First verify basic imports
        print("✅ Triton imported successfully")


        # Test configuration
        n_elements = 1024
        x = torch.randn(n_elements, device='cuda')
        y = torch.randn(n_elements, device='cuda')
        output = torch.empty_like(x)
        BLOCK_SIZE = 256

        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        # Verification
        assert torch.allclose(output, x + y)
        print("✅ Triton kernel executed successfully")
        return None
        
    except Exception as e:
        return f"❌ Triton test failed: {str(e)}"


import warnings

def test_xformers() -> Optional[str]:
    """Test if xformers is properly installed and functional."""
    try:
        # First verify basic imports
        
        from xformers.components.attention.core import scaled_dot_product_attention



        print("✅ xformers imported successfully")


            
        with warnings.catch_warnings(action="ignore"):
            warnings.simplefilter("ignore")

            b, s, d = 4, 90, 16
            prob = 0.8  # make sure that we trigger the sparse kernels

            a = torch.rand(b, s, d)
            mask = torch.rand(b, s, s) > prob

            # mask of bools
            r_dense_bool = scaled_dot_product_attention(a, a, a, mask)
            r_sparse_bool = scaled_dot_product_attention(a, a, a, mask.to_sparse())
            assert torch.allclose(r_dense_bool, r_sparse_bool)

            # Test additive mask. Mask of 0's and -infs.
            float_mask_add = torch.zeros_like(mask, dtype=torch.float)
            float_mask_add = float_mask_add.masked_fill(mask, float("-inf"))

            r_dense_add = scaled_dot_product_attention(a, a, a, float_mask_add)
            r_sparse_add = scaled_dot_product_attention(a, a, a, float_mask_add.to_sparse())

            # Now properly handled
            assert torch.allclose(r_dense_add, r_sparse_add)

            # Test additive mask with mismatched batch dim
            d = b // 2
            mask = torch.rand(d, s, s) > prob
            float_mask_add = torch.zeros_like(mask, dtype=torch.float)
            float_mask_add = float_mask_add.masked_fill(mask, float("-inf"))

            # Make sure masking doesn't return errors
            r_dense_add = scaled_dot_product_attention(a, a, a, float_mask_add)

            
        print("✅ Triton kernel executed successfully")
        return None
        
    except Exception as e:
        return f"❌ Triton test failed: {str(e)}"







def test_sage_attentionx() -> Optional[str]:
    """Test if SparseAttention (SAGE) is properly installed and functional."""
    try:
        sparse_attn = importlib.import_module("sageattention")
        print("✅ SageAttention imported successfully")

        from sageattention import sageattn
        
        # Test with dummy inputs
        batch_size, seq_len, n_heads, d_head = 2, 256, 4, 64
        q = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda').to(torch.float16)
        k = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda').to(torch.float16)
        v = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda').to(torch.float16)
        
        
        

        
        try:
            # This is a placeholder - actual API may vary
            attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
            print("✅ SageAttention forward pass successful")
            return None
        except Exception as e:
            return f"❌ SageAttention forward pass failed: {str(e)}"
    except Exception as e:
        return f"❌ SageAttention import failed: {str(e)}"





def test_sage_attention() -> Optional[str]:
    """Test if SparseAttention (SAGE) is properly installed and functional."""
    try:

        from flash_attn.utils.benchmark import benchmark_forward

        import sageattention._qattn_sm89 as qattn
        from sageattention import sageattn

        print("✅ SageAttention imported successfully")

                    
        quant_gran="per_warp"
        pv_accum_dtype="fp32"
        head = 32
        batch = 32
        headdim = 128


        
        try:
            
            

            WARP_Q = 32
            WARP_K = 64

            if pv_accum_dtype == 'fp32':
                kernel = qattn.qk_int8_sv_f8_accum_f32_attn # the kernel with fp32 (actually fp22) accumulator
            elif pv_accum_dtype == 'fp32+fp32':
                kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf # the kernel with fp32 longterm buffer and fp32 (actually fp22) shortterm accumulator

            _qk_quant_gran = 3 if quant_gran == 'per_thread' else 2

            is_causal = False
            _is_causal = 1 if is_causal else 0
            #for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
            for seq_len in {1024}:
                flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

                q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
                k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
                o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")

                vm = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")
                v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

                if quant_gran == 'per_warp':
                    q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
                    k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
                elif quant_gran == 'per_thread':
                    q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
                    k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

                v = torch.randn(batch, headdim,head,  seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
                sm_scale = 1 / (headdim ** 0.5)
                for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
                torch.cuda.synchronize()
                _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
                #print(f'{seq_len} flops:{flops/time.mean*1e-12}')

            is_causal = True
            _is_causal = 1 if is_causal else 0
            #for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
            for seq_len in {1024}:
                flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)

                q = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
                k = torch.randint(-95, 95, (batch, seq_len, head, headdim), dtype=torch.int8, device="cuda")
                o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")

                vm = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")
                v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

                if quant_gran == 'per_warp':
                    q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float, device="cuda")
                    k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float, device="cuda")
                elif quant_gran == 'per_thread':
                    q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float, device="cuda")
                    k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float, device="cuda")

                v = torch.randn(batch, headdim,head,  seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
                sm_scale = 1 / (headdim ** 0.5)
                for i in range(5): kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
                torch.cuda.synchronize()
                _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
                #print(f'{seq_len} flops:{flops/time.mean*1e-12}')
            

            # This is a placeholder - actual API may vary
            print("✅ SageAttention CUDA QK Int8 PV FP8 Benchmark successful")
            return None
        except Exception as e:
            return f"❌ SageAttention CUDA QK Int8 PV FP8 Benchmark failed: {str(e)}"
    except Exception as e:
        return f"❌ SageAttention import failed: {str(e)}"





def test_causal_conv1d() -> Optional[str]:
    """Test if CausalConv1D is properly installed and functional."""
    try:
        causal_conv1d = importlib.import_module("causal_conv1d")
        print("✅ CausalConv1D imported successfully")
        
        

        import torch
        import torch.nn.functional as F
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref
        from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states, causal_conv1d_varlen_states_ref

        
        itype = torch.float16 
        silu_activation = False
        has_bias=False
        has_cache_seqlens=False
        seqlen=  4
        width=2 #, 3, 4
        dim= 2048# 2048 + 16, 4096

        try:    
            
            
            device = "cuda"
            rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
            if itype == torch.bfloat16:
                rtol, atol = 1e-2, 5e-2
            rtolw, atolw = (1e-3, 1e-3)
            # set seed
            torch.random.manual_seed(0)
            batch = 64
            # batch = 1
            # dim = 64
            x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
            state_len = torch.randint(width - 1, width + 10, (1,)).item()

            total_entries = 10 * batch
            conv_state = torch.randn(total_entries, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
            conv_state_indices = torch.randperm(total_entries)[:batch].to(dtype=torch.int32, device=device)

            weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
            if has_bias:
                bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
            else:
                bias = None
            conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
            activation = None if not silu_activation else "silu"
            cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                            if has_cache_seqlens else None)
            out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation,
                                    cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices)
            out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

            assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
            print("✅ CausalConv1D forward pass successful")
            return None
        except Exception as e:
            return f"❌ CausalConv1D forward pass failed: {str(e)}"
    except Exception as e:
        return f"❌ CausalConv1D import failed: {str(e)}"

def test_mamba() -> Optional[str]:
    """Test if Mamba is properly installed and functional."""
    try:
        


        import torch
        import torch.nn.functional as F

        from einops import rearrange

        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_ref


        print("✅ Mamba imported successfully")
                                

        # wtype=torch.float32, torch.complex64])
        wtype=torch.float32
        # itype=torch.float32, torch.float16, torch.bfloat16])
        itype=torch.float32
        # seqlen=8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
        seqlen=128
        # seqlen=128])
        # return_last_state=False, True])
        return_last_state=True
        # has_delta_bias=False, True])
        has_delta_bias=True
        # delta_softplus=False, True])
        delta_softplus=True
        # has_z=False, True])
        has_z=True
        # has_D=False, True])
        has_D=True
        varBC_groups=1 #1 ,2
        # varBC_groups=1])
        # is_variable_C=False, True])
        is_variable_C=True
        # is_variable_B=False, True])
        is_variable_B=True
        #ää---------TEST START------------------
        
        
        
          
        try:
            
            if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
                print ("TEST NOT APPLICABLE")
                assert False  # This config is not applicable
            device = 'cuda'
            rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
            if itype == torch.bfloat16:
                rtol, atol = 3e-2, 5e-2
            rtolw, atolw = (1e-3, 1e-3)
            if has_z:  # If we have z, the errors on the weights seem higher
                rtolw = max(rtolw, rtol)
                atolw = max(atolw, atol)
            # set seed
            torch.random.manual_seed(0)
            batch_size = 2
            dim = 4
            dstate = 8
            is_complex = wtype == torch.complex64
            A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
            if not is_variable_B:
                B_shape = (dim, dstate)
            elif varBC_groups == 1:
                B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
            else:
                B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
            B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                            requires_grad=True)
            if not is_variable_C:
                C_shape = (dim, dstate)
            elif varBC_groups == 1:
                C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
            else:
                C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
            C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                            requires_grad=True)
            if has_D:
                D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
            else:
                D = None
            if has_z:
                z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
            else:
                z = None
            if has_delta_bias:
                delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
            else:
                delta_bias = None
            u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
            delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
            A_ref = A.detach().clone().requires_grad_()
            B_ref = B.detach().clone().requires_grad_()
            C_ref = C.detach().clone().requires_grad_()
            D_ref = D.detach().clone().requires_grad_() if D is not None else None
            z_ref = z.detach().clone().requires_grad_() if z is not None else None
            u_ref = u.detach().clone().requires_grad_()
            delta_ref = delta.detach().clone().requires_grad_()
            delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
            out, *rest = selective_scan_fn(
                u, delta, A, B, C, D, z=z,
                delta_bias=delta_bias, delta_softplus=delta_softplus,
                return_last_state=return_last_state
            )
            if return_last_state:
                state = rest[0]
            out_ref, *rest = selective_scan_ref(
                u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
                delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
                return_last_state=return_last_state
            )
            if return_last_state:
                state_ref = rest[0]
            # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
            # dt_u = delta * u

            assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
            if return_last_state:
                assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

            g = torch.randn_like(out)
            out_ref.backward(g)
            out.backward(g)


            assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
            assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
            assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
            assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                                atol=atolw if not is_variable_B else atol)
            assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                                atol=atolw if not is_variable_C else atol)
            if has_D:
                assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
            if has_z:
                assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
            if has_delta_bias:
                assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)


              


            print("✅ Mamba selective_scan successful")
            return None
        except Exception as e:
            return f"❌ Mamba forward pass failed: {str(e)}"
    except Exception as e:
        return f"❌ Mamba import failed: {str(e)}"




def test_flash_attention() -> Optional[str]:
    """Test if FlashAttention is properly installed and functional."""
    try:
        import torch
        flash_attn = importlib.import_module("flash_attn")
        print("✅ FlashAttention imported successfully")
        
        # Test FlashAttention with dummy inputs
        batch_size, seq_len, n_heads, d_head = 2, 256, 4, 64
        dtype = torch.float16
        
        # Create inputs with proper stride order for FlashAttention
        q = torch.randn(batch_size, n_heads, seq_len, d_head, 
                       device='cuda', dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, 
                       device='cuda', dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, 
                       device='cuda', dtype=dtype)
        
        try:
            output = flash_attn.flash_attn_func(
                q, k, v, 
                causal=True,  # Usually want causal for most applications
                softmax_scale=None  # Default scale
            )
            print("✅ FlashAttention forward pass successful")
            return None
        except Exception as e:
            return f"❌ FlashAttention forward pass failed: {str(e)}"
    except Exception as e:
        return f"❌ FlashAttention import failed: {str(e)}"


def main():
    print_header("Starting Deep Learning Libraries Test")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available - tests will fail")
        return
    
    tests = [
        ("Triton", test_triton),
   #     ("xformers", test_xformers),
        ("FlashAttention", test_flash_attention),
        ("SageAttention", test_sage_attention),
        ("CausalConv1D", test_causal_conv1d),
        ("Mamba", test_mamba),
    ]
    

    
    
    for i, (name, test_fn) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Testing {name}...")
        if error := test_fn():
            print(error)
    
    print_header("Testing complete")

if __name__ == "__main__":
    main()