import numpy as np


def test_matmul_2d_numba(init_cuda_simulator):
    # this needs to be imported after the simulator is initialized
    from cudagrad.kernels.matmul import matmul_2d_numba

    matrix_a = np.random.randn(32, 32).astype(np.float32)
    matrix_b = np.random.randn(32, 32).astype(np.float32)

    output_matrix = matmul_2d_numba(matrix_a, matrix_b)
    expected_output = np.matmul(matrix_a, matrix_b)

    np.testing.assert_allclose(output_matrix, expected_output, atol=1e-5)
