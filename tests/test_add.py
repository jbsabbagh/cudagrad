import numpy as np


def test_add_2d_numba(init_cuda_simulator):
    # this needs to be imported after the simulator is initialized
    from cudagrad.kernels.add import add_2d_numba
    from numba import cuda

    matrix_a = cuda.to_device(np.random.randn(32, 32).astype(np.float32))
    matrix_b = cuda.to_device(np.random.randn(32, 32).astype(np.float32))

    output_matrix = add_2d_numba(matrix_a, matrix_b)
    expected_output = np.add(matrix_a, matrix_b)

    np.testing.assert_allclose(output_matrix, expected_output, atol=1e-5)


def test_add_with_scalar_2d_numba(init_cuda_simulator):
    from cudagrad.kernels.add import add_scalar_to_2d_matrix_numba
    from numba import cuda

    matrix_a = cuda.to_device(np.random.randn(32, 32).astype(np.float32))
    scalar = np.random.randn(1).astype(np.float32)

    output_matrix = add_scalar_to_2d_matrix_numba(matrix_a, scalar)
    expected_output = np.add(matrix_a, scalar)

    np.testing.assert_allclose(output_matrix, expected_output, atol=1e-5)
