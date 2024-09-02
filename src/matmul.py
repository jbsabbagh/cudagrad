import torch
import numpy as np
from numba import cuda
from numba.cuda import as_cuda_array as ca
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

    shared_memory = cuda.shared.array(0, dtype=np.float32)
    shared_matrix_a, shared_matrix_b = (
        shared_memory[: tile_width * tile_width],
        shared_memory[tile_width * tile_width : 2 * tile_width * tile_width],
    )

    partial_sum = np.float32(0.0)
    for phase in range(math.ceil(width_a / tile_width)):
        index = phase * tile_width
        shared_matrix_a[thread_row * tile_width + thread_col] = (
            matrix_a[row, thread_col + index]
            if row < height_a and index + thread_col < width_a
            else 0.0
        )
        shared_matrix_b[thread_row * tile_width + thread_col] = (
            matrix_b[thread_row + index, col]
            if col < width_b and index + thread_row < width_a
            else 0.0
        )
        cuda.syncthreads()

        for i in range(tile_width):
            partial_sum += (
                shared_matrix_a[thread_row * tile_width + i]
                * shared_matrix_b[i * tile_width + thread_col]
            )
        cuda.syncthreads()
    if row < height_a and col < width_b:
        output_matrix[row, col] = partial_sum


def matmul_2d_numba(matrix_a, matrix_b, tile_width=16):
    height_a, width_a = matrix_a.shape
    width_b, height_b = matrix_b.shape
    assert width_a == width_b, "Size mismatch!"
    output_matrix = torch.zeros(
        height_a, height_b, dtype=matrix_a.dtype, device=matrix_a.device
    )
    dynamic_shared_memory_size = 2 * tile_width * tile_width * 4
    threads_per_block = tile_width, tile_width
    blocks_per_grid = (
        cdiv(height_b, threads_per_block[0]),
        cdiv(height_a, threads_per_block[1]),
    )
    matmul_k_numba[blocks_per_grid, threads_per_block, 0, dynamic_shared_memory_size](
        ca(matrix_a), ca(matrix_b), ca(output_matrix), tile_width
    )
    return output_matrix
