from cudagrad.matmul import matmul_2d_numba


def test_matmul_2d_numba():
    import numpy as np

    # Initialize input matrices
    matrix_a = np.random.randn(32, 32).astype(np.float32)
    matrix_b = np.random.randn(32, 32).astype(np.float32)

    # Perform matrix multiplication using the matmul_2d_numba function
    output_matrix = matmul_2d_numba(matrix_a, matrix_b)

    # Perform matrix multiplication using NumPy for verification
    expected_output = np.matmul(matrix_a, matrix_b)

    # Check if the output from matmul_2d_numba is close to the expected output
    assert np.allclose(output_matrix, expected_output, atol=1e-5), "Test failed!"
