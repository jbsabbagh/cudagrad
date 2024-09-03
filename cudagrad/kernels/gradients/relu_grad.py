from numba import cuda
from cudagrad.kernels.utils import cdiv
from cudagrad.kernels.constants import WARP_SIZE


@cuda.jit
def _relu_gradient_numba_kernel(a, grad, output):
    block_idx, block_dim, thread_idx = cuda.blockIdx, cuda.blockDim, cuda.threadIdx
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x

    if row < a.shape[0] and col < a.shape[1]:
        mask = 1 if a[row, col] > 0 else 0
        output[row, col] = grad[row, col] * mask


def relu_gradient_numba(a, grad):
    output = cuda.device_array_like(a)
    threads_per_block = (WARP_SIZE, WARP_SIZE)
    blocks_per_grid = (
        cdiv(a.shape[0], threads_per_block[0]),
        cdiv(a.shape[1], threads_per_block[1]),
    )
    _relu_gradient_numba_kernel[blocks_per_grid, threads_per_block](a, grad, output)
    return output
