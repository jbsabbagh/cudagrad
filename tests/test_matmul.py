import numpy as np
from cudagrad.kernels.matmul import matmul_2d_numba
from numba import cuda


def test_matmul_2d_numba():
    matrix_a = cuda.to_device(np.random.randn(32, 32).astype(np.float32))
    matrix_b = cuda.to_device(np.random.randn(32, 32).astype(np.float32))

    output_matrix = matmul_2d_numba(matrix_a, matrix_b)
    expected_output = np.matmul(matrix_a, matrix_b)

    np.testing.assert_allclose(output_matrix, expected_output, atol=1e-5)
