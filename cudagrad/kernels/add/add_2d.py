import numpy as np
from numba import cuda
from cudagrad.kernels.utils import cdiv

@cuda.jit
def _add_1d_numba_kernel(a, b, output):
    block_idx, block_dim, thread_idx = cuda.blockIdx, cuda.blockDim, cuda.threadIdx
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x

    if row < a.shape[0] and col < a.shape[1]:
        output[row, col] = a[row, col] + b[row, col]

def add_2d_numba(a, b):
    output = cuda.device_array_like(a)
    threads_per_block = (16, 16)
    blocks_per_grid = (cdiv(a.shape[0], threads_per_block[0]), cdiv(a.shape[1], threads_per_block[1]))
    _add_1d_numba_kernel[blocks_per_grid, threads_per_block](a, b, output)
    return output
