import numpy as np
from numba import cuda
import math


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a + b - 1) // b


@cuda.jit
def matmul_k_numba(matrix_a, matrix_b, output_matrix, tile_width):
    block_idx, block_dim, thread_idx = cuda.blockIdx, cuda.blockDim, cuda.threadIdx
    thread_col, thread_row = thread_idx.x, thread_idx.y

    row = block_idx.y * block_dim.y + thread_row
    col = block_idx.x * block_dim.x + thread_col

    height_a, width_a = matrix_a.shape
    height_b, width_b = matrix_b.shape

    partial_sum = np.float32(0.0)
    if row < height_a and col < width_b:
        for k in range(width_a):
            partial_sum += matrix_a[row, k] * matrix_b[k, col]

        output_matrix[row, col] = partial_sum


def matmul_2d_numba(matrix_a, matrix_b, tile_width=16):
    height_a, width_a = matrix_a.shape
    height_b, width_b = matrix_b.shape
    assert width_a == height_b, "Size mismatch!"
    output_matrix = cuda.device_array((height_a, width_b), dtype=matrix_a.dtype)
    threads_per_block = (tile_width, tile_width)
    blocks_per_grid = (
        cdiv(width_b, threads_per_block[0]),
        cdiv(height_a, threads_per_block[1]),
    )
    matmul_k_numba[blocks_per_grid, threads_per_block](
        matrix_a, matrix_b, output_matrix, tile_width
    )
    return output_matrix
