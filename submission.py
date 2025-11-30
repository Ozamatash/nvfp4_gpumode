import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
from cutlass.cute import TensorSSA
from cutlass.cute.runtime import make_ptr
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Float16, Float8E4M3FN, Int8, Int16, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm, arith, vector, builtin
from cutlass._mlir import ir

# Kernel configuration parameters
ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E4M3FN  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type
sf_vec_size = 16  # Scale factor block size (16 elements share one scale)
threads_per_m = 16  # Number of threads per CUDA thread block
threads_per_k  = 16
mma_tiler_mnk = (threads_per_m, 1, 128)  # Tile sizes for M, N, K dimensions
accum_dtype = cutlass.Float32  # Float32 accumulation buffer


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# ============================================================================
# FP8 (E4M3) to FP16 conversion intrinsics
# These provide optimized conversion using PTX cvt.rn.f16x2.e4m3x2 instructions
# ============================================================================

@dsl_user_op
def cvt_f8e4m3x2_to_f16x2(src_vec2, *, loc=None, ip=None):
    """Convert 2 float8e4m3 values to 2 float16 values"""
    # pack 2 float8e4m3 into 1 int16 value
    src_i16 = llvm.bitcast(Int16.mlir_type, src_vec2, loc=loc, ip=ip)
    rst_i32 = llvm.inline_asm(
        Int32.mlir_type,
        [src_i16],
        """{\n\t
            cvt.rn.f16x2.e4m3x2 $0, $1;\n\t
        }""",
        "=r,h",
    )
    vec_f16x2_type = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    vec_f16x2 = llvm.bitcast(vec_f16x2_type, rst_i32, loc=loc, ip=ip)
    return vec_f16x2


@dsl_user_op
def cvt_f8e4m3_f16(src, *, loc=None, ip=None):
    """Convert single float8e4m3 value to float16"""
    # 0 padding for upper 8 bits
    zero = arith.constant(src.type, 0, loc=loc, ip=ip)
    vec2 = vector.from_elements(
        ir.VectorType.get([2], src.type, loc=loc), [src, zero], loc=loc, ip=ip
    )
    rst_vec2 = cvt_f8e4m3x2_to_f16x2(vec2, loc=loc, ip=ip)
    # only the 1st element is valid
    rst = vector.extract(
        rst_vec2, dynamic_position=[], static_position=[0], loc=loc, ip=ip
    )
    return rst


@dsl_user_op
def cvt_f8e4m3x4_to_f16x4(src_vec4, *, loc=None, ip=None):
    """Convert 4 float8e4m3 values to 4 float16 values"""
    # pack 4 float8e4m3 into 1 int32 value
    src_i32 = llvm.bitcast(Int32.mlir_type, src_vec4, loc=loc, ip=ip)
    rst_i32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [src_i32],
        """{\n\t
            .reg .b16 h0, h1;\n\t
            mov.b32 {h0, h1}, $2;\n\t
            cvt.rn.f16x2.e4m3x2 $0, h0;\n\t
            cvt.rn.f16x2.e4m3x2 $1, h1;\n\t
        }""",
        "=r,=r,r",
    )
    res0 = llvm.extractvalue(T.i32(), rst_i32x2, [0])
    res1 = llvm.extractvalue(T.i32(), rst_i32x2, [1])
    vec_i32x2_type = ir.VectorType.get([2], Int32.mlir_type, loc=loc)
    vec_i32x2 = vector.from_elements(vec_i32x2_type, [res0, res1], loc=loc, ip=ip)
    vec_f16x4_type = ir.VectorType.get([4], Float16.mlir_type, loc=loc)
    vec_f16x4 = llvm.bitcast(vec_f16x4_type, vec_i32x2, loc=loc, ip=ip)
    return vec_f16x4


@dsl_user_op
def cvt_f8e4m3x8_to_f16x8(src_vec8, *, loc=None, ip=None):
    """Convert 8 float8e4m3 values to 8 float16 values"""
    # Split into two i32 values instead of using i64
    vec_i32x2_type = ir.VectorType.get([2], Int32.mlir_type, loc=loc)
    src_i32x2 = llvm.bitcast(vec_i32x2_type, src_vec8, loc=loc, ip=ip)
    src_lo = llvm.extractelement(src_i32x2, arith.constant(Int32.mlir_type, 0), loc=loc, ip=ip)
    src_hi = llvm.extractelement(src_i32x2, arith.constant(Int32.mlir_type, 1), loc=loc, ip=ip)
    
    # Process lower 4 bytes (4 fp8 values)
    rst_lo_i32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [src_lo],
        """{\n\t
            .reg .b16 h0, h1;\n\t
            mov.b32 {h0, h1}, $2;\n\t
            cvt.rn.f16x2.e4m3x2 $0, h0;\n\t
            cvt.rn.f16x2.e4m3x2 $1, h1;\n\t
        }""",
        "=r,=r,r",
    )
    
    # Process upper 4 bytes (4 fp8 values)
    rst_hi_i32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [src_hi],
        """{\n\t
            .reg .b16 h0, h1;\n\t
            mov.b32 {h0, h1}, $2;\n\t
            cvt.rn.f16x2.e4m3x2 $0, h0;\n\t
            cvt.rn.f16x2.e4m3x2 $1, h1;\n\t
        }""",
        "=r,=r,r",
    )
    
    res0 = llvm.extractvalue(T.i32(), rst_lo_i32x2, [0])
    res1 = llvm.extractvalue(T.i32(), rst_lo_i32x2, [1])
    res2 = llvm.extractvalue(T.i32(), rst_hi_i32x2, [0])
    res3 = llvm.extractvalue(T.i32(), rst_hi_i32x2, [1])
    
    vec_i32x4_type = ir.VectorType.get([4], Int32.mlir_type, loc=loc)
    vec_i32x4 = vector.from_elements(
        vec_i32x4_type, [res0, res1, res2, res3], loc=loc, ip=ip
    )
    vec_f16x8_type = ir.VectorType.get([8], Float16.mlir_type, loc=loc)
    vec_f16x8 = llvm.bitcast(vec_f16x8_type, vec_i32x4, loc=loc, ip=ip)
    return vec_f16x8


@dsl_user_op
def cvt_f8e4m3_f16_intrinsic(vec_f8e4m3, length, *, loc=None, ip=None):
    """
    Convert a vector of float8e4m3 to a vector of float16.
    
    :param vec_f8e4m3: The input vector of float8e4m3.
    :param length: The length of the input vector.
    :return: The output 1D vector of float16 with the same length as the input vector.
    """
    src_pos = 0
    vec_src_i8 = builtin.unrealized_conversion_cast(
        [ir.VectorType.get([length], Int8.mlir_type, loc=loc)],
        [vec_f8e4m3],
        loc=loc,
        ip=ip,
    )
    vec_i8x8_type = ir.VectorType.get([8], Int8.mlir_type, loc=loc)
    vec_i8x4_type = ir.VectorType.get([4], Int8.mlir_type, loc=loc)
    vec_i8x2_type = ir.VectorType.get([2], Int8.mlir_type, loc=loc)
    vec_dst_type = ir.VectorType.get([length], Float16.mlir_type, loc=loc)
    vec_dst = llvm.mlir_zero(vec_dst_type, loc=loc, ip=ip)

    # try to use vectorized version
    if length >= 8:
        num_vec8 = length // 8
        for _ in range(num_vec8):
            vec_f8e4m3x8 = vector.extract_strided_slice(
                vec_i8x8_type, vec_src_i8, [src_pos], [8], [1], loc=loc, ip=ip
            )
            vec_f16x8 = cvt_f8e4m3x8_to_f16x8(vec_f8e4m3x8, loc=loc, ip=ip)
            vec_dst = vector.insert_strided_slice(
                vec_f16x8, vec_dst, [src_pos], [1], loc=loc, ip=ip
            )
            src_pos += 8
            length -= 8

    if length >= 4:
        vec_f8e4m3x4 = vector.extract_strided_slice(
            vec_i8x4_type, vec_src_i8, [src_pos], [4], [1], loc=loc, ip=ip
        )
        vec_f16x4 = cvt_f8e4m3x4_to_f16x4(vec_f8e4m3x4, loc=loc, ip=ip)
        vec_dst = vector.insert_strided_slice(
            vec_f16x4, vec_dst, [src_pos], [1], loc=loc, ip=ip
        )
        src_pos += 4
        length -= 4

    if length >= 2:
        vec_f8e4m3x2 = vector.extract_strided_slice(
            vec_i8x2_type, vec_src_i8, [src_pos], [2], [1], loc=loc, ip=ip
        )
        vec_f16x2 = cvt_f8e4m3x2_to_f16x2(vec_f8e4m3x2, loc=loc, ip=ip)
        vec_dst = vector.insert_strided_slice(
            vec_f16x2, vec_dst, [src_pos], [1], loc=loc, ip=ip
        )
        src_pos += 2
        length -= 2

    if length >= 1:
        val_f16 = cvt_f8e4m3_f16(
            vector.extractelement(
                vec_src_i8,
                position=arith.constant(Int32.mlir_type, src_pos),
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )
        vec_dst = vector.insertelement(
            val_f16,
            vec_dst,
            position=arith.constant(Int32.mlir_type, src_pos),
            loc=loc,
            ip=ip,
        )

    return vec_dst

@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    # Get CUDA block and thread indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, _ = cute.arch.thread_idx()

    # Extract the local tile for input matrix A (shape: [block_M, block_K, rest_M, rest_K, rest_L])
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for A (same shape as gA_mkl)
    # Here, block_M = (32, 4); block_K = (16, 4)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for input matrix B (shape: [block_N, block_K, rest_N, rest_K, rest_L])
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for B (same shape as gB_nkl)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for output matrix C (shape: [block_M, block_N, rest_M, rest_N, rest_L])
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    # Select output element corresponding to this thread and block indices
    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.make_rmem_tensor_like(cute.make_layout(2), c_dtype)
    res.fill(0)

    # Shared Memory
    allocator = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((threads_per_m, threads_per_k), stride = (threads_per_k, 1))
    shared_res = allocator.allocate_tensor(element_type=c_dtype, layout=smem_layout)

    # Get the number of k tiles (depth dimension) for the reduction loop
    k_tile_cnt = gA_mkl.layout[3].shape
    for k_tile in range(tidy, k_tile_cnt, threads_per_k, unroll_full=True):
        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[0, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[0, None, bidy, k_tile, bidz]

        tArA = cute.make_rmem_tensor_like(tAgA, c_dtype)
        tBrB = cute.make_rmem_tensor_like(tBgB, c_dtype)
        tABrAB = cute.make_rmem_tensor_like(tAgA, c_dtype)
        tArSFA = cute.make_rmem_tensor_like(tAgSFA, accum_dtype)
        tBrSFB = cute.make_rmem_tensor_like(tBgSFB, accum_dtype)
        tSFrSF = cute.make_rmem_tensor_like(tAgSFA, accum_dtype)

        # Load NVFP4 or FP8 values from global memory
        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        # Convert FP4 -> FP16 and FP8 -> FP16
        sfa_length = cute.size(tAgSFA.layout)
        sfb_length = cute.size(tBgSFB.layout)
        a_val = a_val_nvfp4.to(c_dtype)
        b_val = b_val_nvfp4.to(c_dtype)
        sfa_val_Vec = cvt_f8e4m3_f16_intrinsic(sfa_val_fp8, sfa_length)
        sfa_val = cute.TensorSSA(sfa_val_Vec, tAgSFA.layout.shape, c_dtype)
        sfb_val_Vec = cvt_f8e4m3_f16_intrinsic(sfb_val_fp8, sfb_length)
        sfb_val = cute.TensorSSA(sfb_val_Vec, tBgSFB.layout.shape, c_dtype)

        # Store the converted values to RMEM CuTe tensors
        tArA.store(a_val)
        tBrB.store(b_val)
        tArSFA.store(sfa_val)
        tBrSFB.store(sfb_val)

        tABrAB.store(tArA.load() * tBrB.load())
        tSFrSF.store(tArSFA.load() * tBrSFB.load())

        # Iterate over SF vector tiles and compute the scale&matmul accumulation
        for i in cutlass.range_constexpr(mma_tiler_mnk[2]):
            res += tABrAB[i] * tSFrSF[i]
   
    shared_res[(tidx, tidy)] = res[0]
    cute.arch.sync_threads()
    
    if tidy == 0:
        out = cute.zeros_like(tCgC, accum_dtype)
        for i in cutlass.range_constexpr(threads_per_k):
            out += shared_res[(tidx, i)]

        # Store the final float16 result back to global memory
        tCgC.store(out.to(cutlass.Float16))
    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, _, k, l = problem_size
    # Create CuTe Tensor via pointer and problem size.
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    # We use n=128 to create the torch tensor to do fp4 computation via torch._scaled_mm
    # then copy torch tensor to cute tensor for cute customize kernel computation
    # therefore we need to ensure b_tensor has the right stride with this 128 padded size on n.
    n_padded_128 = 128
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n_padded_128, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n_padded_128 * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), 1, l), stride=(1, 1, m))
    )
    # Convert scale factor tensors to MMA layout
    # The layout matches Tensor Core requirements: (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

    # Compute grid dimensions
    # Grid is (M_blocks, 1, L) where:
    # - M_blocks = ceil(M / 128) to cover all output rows
    # - L = batch size
    grid = (
        cute.ceil_div(c_tensor.shape[0], threads_per_m),
        1,
        c_tensor.shape[2],
    )

    # Launch the CUDA kernel
    kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=grid,
        block=[threads_per_m, threads_per_k, 1],
        cluster=(1, 1, 1),
    )
    return


# Global cache for compiled kernel
_compiled_kernel_cache = None


# This function is used to compile the kernel once and cache it and then allow users to
# run the kernel multiple times to get more accurate timing results.
def compile_kernel():
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache

    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(
        my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
    )

    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.

    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.

    Args:
        data: Tuple of (a, b, sfa_cpu, sfb_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn
            b: [1, k, l] - Input vector in float4e2m1fn
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn
            sfb_cpu: [1, k, l] - Scale factors in float8_e4m3fn
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, 1, l] - Output vector in float16

    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2
    # GEMV N dimension is always 1
    n = 1

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c