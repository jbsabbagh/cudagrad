from numba import cuda
from cudagrad.kernels.utils import cdiv
from cudagrad.kernels.constants import WARP_SIZE


@cuda.jit
def _power_2d_numba_kernel(a, output, power):
    block_idx, block_dim, thread_idx = cuda.blockIdx, cuda.blockDim, cuda.threadIdx
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x

    if row < a.shape[0] and col < a.shape[1]:
        output[row, col] = a[row, col] ** power


def power_2d_numba(a, power):
    output = cuda.device_array_like(a)
    threads_per_block = (WARP_SIZE, WARP_SIZE)
    blocks_per_grid = (
        cdiv(a.shape[0], threads_per_block[0]),
        cdiv(a.shape[1], threads_per_block[1]),
    )

    _power_2d_numba_kernel[blocks_per_grid, threads_per_block](a, output, power)

    return output
