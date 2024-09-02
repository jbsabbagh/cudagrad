from numba import cuda
from cudagrad.kernels.utils import cdiv
from cudagrad.kernels.constants import WARP_SIZE

@cuda.jit
def _add_with_scalar_2d_numba_kernel(matrix, scalar, output):
    block_idx, block_dim, thread_idx = cuda.blockIdx, cuda.blockDim, cuda.threadIdx
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x

    if row < matrix.shape[0] and col < matrix.shape[1]:
        output[row, col] = matrix[row, col] + scalar

def add_scalar_to_2d_matrix_numba(matrix, scalar):
    output = cuda.device_array_like(matrix)
    threads_per_block = (WARP_SIZE, WARP_SIZE)
    blocks_per_grid = (cdiv(matrix.shape[0], threads_per_block[0]), cdiv(matrix.shape[1], threads_per_block[1]))
    _add_with_scalar_2d_numba_kernel[blocks_per_grid, threads_per_block](matrix, scalar, output)
    return output
