from typing import Callable, Optional

from . import operators

from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor import Tensor
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
MAX_THREADS = 1024
# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
        """
        Perform the forward pass of attention softmax using a custom CUDA kernel.

        Args:
            inp  : Input tensor representing attention scores 
                of shape (batch_size, nhead, from_len, to_len)
            mask : Mask tensor applied before softmax, 
                of shape (batch_size, nhead, from_len, to_len)

        Returns:
            inp  : The input tensor after applying softmax in-place
        """
        # Step 1: Retrieve input shape
        batch_size, nhead, from_len, to_len = inp.shape
        is_dec_self_attn = False # Flag for decoder self-attention
        stream = torch.cuda.current_stream().cuda_stream # Get the current CUDA stream

        # Step 2: Define argument types for the CUDA kernel
        lib_softmax.launch_attn_softmax.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # Input tensor (attention scores)
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # Mask tensor
        ctypes.c_int, # Batch size
        ctypes.c_int, # Number of attention heads
        ctypes.c_int, # Source sequence length (from_len)
        ctypes.c_int, # Target sequence length (to_len)
        ctypes.c_bool, # Flag for decoder self-attention
        ctypes.c_void_p # CUDA stream
        ]
        lib_softmax.launch_attn_softmax.restype = None

        # Step 3: Call the CUDA kernel to perform softmax operation on attention scores
        lib_softmax.launch_attn_softmax(
        inp._tensor._storage, # Input tensor (attention scores)
        mask._tensor._storage, # Mask tensor
        batch_size, # Batch size
        nhead, # Number of attention heads
        from_len, # Source sequence length
        to_len, # Target sequence length
        is_dec_self_attn, # Boolean flag for decoder self-attention
        stream # CUDA stream for parallel execution
        ) 
        # Step 4: Return the modified input tensor after applying softmax
        return inp

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
        """
        Perform the backward pass of attention softmax using a custom CUDA kernel.

        Args:
            out_grad : Gradient of the output tensor from the next layer, 
                    of shape (batch_size, nhead, from_len, to_len)
            soft_inp : Softmax output tensor from the forward pass, 
                    of shape (batch_size, nhead, from_len, to_len)

        Returns:
            out_grad : The modified gradient tensor after applying softmax backward computation.
        """
        #   BEGIN ASSIGN3_1
        # Step 1: Get the current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream

        # Step 2: Retrieve input shape information
        batch_size = soft_inp.shape[0] # Number of batches
        nhead = soft_inp.shape[1] # Number of attention heads
        from_len = soft_inp.shape[2] # Source sequence length
        to_len = soft_inp.shape[3] # Target sequence length
        int_rows = batch_size * nhead * from_len # Total number of softmax vectors to process

        # Step 3: Define argument types for the CUDA kernel
        lib_softmax.launch_attn_softmax_bw.argtypes = [
            ctypes.c_void_p,  # out_grad pointer
            ctypes.c_void_p,  # soft_inp pointer
            ctypes.c_int,     # rows
            ctypes.c_int,     # softmax_len
            ctypes.c_void_p   # CUDA stream
        ]
        lib_softmax.launch_attn_softmax_bw.restype = None

        # Step 4: Call the CUDA kernel to perform softmax backward pass
        lib_softmax.launch_attn_softmax_bw(
            ctypes.c_void_p(out_grad._tensor._storage.ctypes.data), # Gradient tensor pointer
            ctypes.c_void_p(soft_inp._tensor._storage.ctypes.data), # Softmax output tensor pointer
            int_rows, # Number of softmax rows
            to_len, # Softmax length
            stream # CUDA stream for parallel execution
        )

        # Step 5: Return the modified gradient tensor after backward computation
        return out_grad
        #   END ASSIGN3_1

    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
        """
        Perform Layer Normalization using a custom CUDA kernel.

        Args:
            inp  : Input tensor of shape (batch_size, hidden_dim)
            gamma: Scale parameter (learnable) of shape (hidden_dim,)
            beta : Shift parameter (learnable) of shape (hidden_dim,)

        Returns:
            ln_res: Normalized output tensor of shape (batch_size, hidden_dim)
                    with additional attributes `var` (variance) and `mean`.
        """
        #   BEGIN ASSIGN3_2
        # Step 1: Get input shape and current CUDA stream
        batch_size, hidden_dim = inp.shape
        stream = torch.cuda.current_stream().cuda_stream

        # Step 2: Initialize output tensors for layer norm result, variance, and mean
        ln_res = inp.zeros((batch_size, hidden_dim)) # Output tensor for layer normalization
        vars_  = inp.zeros((batch_size,)) # Variance tensor
        means_ = inp.zeros((batch_size,)) # Mean tensor

        # step 3
        lib_layernorm.launch_layernorm.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # ln_res
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # vars_
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # means_
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # inp
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # gamma
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # beta
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # hidden_dim
            ctypes.c_void_p  # cuda stream
        ]
        lib_layernorm.launch_layernorm.restype = None

        # Step 4: Call the CUDA kernel to compute layer normalization
        lib_layernorm.launch_layernorm(
            ln_res._tensor._storage,
            vars_._tensor._storage,
            means_._tensor._storage,
            inp._tensor._storage,
            gamma._tensor._storage,
            beta._tensor._storage,
            batch_size,
            hidden_dim,
            stream
        )
        # Store variance and mean in the output tensor for potential use in backward pass
        ln_res.var = vars_
        ln_res.mean = means_
        return ln_res
        #   END ASSIGN3_2
      
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, var: Tensor, mean: Tensor):
        """
        Compute the backward pass for Layer Normalization using a custom CUDA kernel.

        Args:
            out_grad: Gradient of the output tensor (batch_size, hidden_dim)
            inp     : Input tensor before normalization (batch_size, hidden_dim)
            gamma   : Scale parameter (learnable) of shape (hidden_dim,)
            beta    : Shift parameter (learnable) of shape (hidden_dim,)
            var     : Variance computed during forward pass (batch_size,)
            mean    : Mean computed during forward pass (batch_size,)

        Returns:
            inp_grad: Gradient of the input tensor (batch_size, hidden_dim)
        """
        #   BEGIN ASSIGN3_2
        # Step 1: Retrieve shape information
        batch_size, hidden_dim = inp.shape
        stream_1 = torch.cuda.current_stream().cuda_stream
        stream_2 = stream_1  # Use the same CUDA stream

        # Step 2: Allocate gradient tensors for gamma, beta, and input
        gamma_grad = inp.zeros((hidden_dim,))
        betta_grad = inp.zeros((hidden_dim,))
        inp_grad   = inp.zeros((batch_size, hidden_dim))

        lib_layernorm.launch_layernorm_bw.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # gamma_grad
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # betta_grad
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # inp_grad
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # out_grad
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # inp
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # gamma
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # beta
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # vars
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # means
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # Hidden dimension
            ctypes.c_void_p,  # CUDA stream 1
            ctypes.c_void_p   # CUDA stream 2
        ]
        lib_layernorm.launch_layernorm_bw.restype = None
        # Step 4: Call the CUDA kernel to compute Layer Normalization backward pass
        lib_layernorm.launch_layernorm_bw(
            gamma_grad._tensor._storage,
            betta_grad._tensor._storage,
            inp_grad._tensor._storage,
            out_grad._tensor._storage,
            inp._tensor._storage,
            gamma._tensor._storage,
            beta._tensor._storage,
            var._tensor._storage,
            mean._tensor._storage,
            batch_size,
            hidden_dim,
            stream_1,
            stream_2
        )

        # Step 5: Return the gradient of the input tensor
        return inp_grad
        #   END ASSIGN3_2