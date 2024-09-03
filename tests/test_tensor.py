import numpy as np
from cudagrad.engine import Tensor
import torch


def test_tensor__init__():
    a = Tensor(np.ones((2, 2)))
    assert a.data.tolist() == [[1, 1], [1, 1]]
    assert a.grad.tolist() == [[0, 0], [0, 0]]
    assert a._prev == set()
    assert a._op == ""


def test_tensor__add__scalar():
    a = Tensor(np.ones((2, 2)))
    b = a + 1
    assert b.data.tolist() == [[2, 2], [2, 2]]
    assert b._op == "+"
    assert a in b._prev


def test_tensor__add__tensor():
    a = Tensor(np.ones((2, 2)))
    b = Tensor(np.ones((2, 2)))
    c = a + b
    assert c.data.tolist() == [[2, 2], [2, 2]]
    assert c._prev == {a, b}
    assert c._op == "+"


def test_tensor_backward__add__():
    # CudaGrad implementation
    a = Tensor(np.ones((2, 2)))
    b = Tensor(np.zeros((2, 2)))
    c = 1 + a + b + a
    c.backward()

    # PyTorch implementation for comparison
    a_torch = torch.ones((2, 2), requires_grad=True)
    b_torch = torch.zeros((2, 2), requires_grad=True)
    c_torch = 1 + a_torch + b_torch + a_torch
    c_torch.sum().backward()

    # Compare gradients
    np.testing.assert_allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(b.grad, b_torch.grad.numpy(), atol=1e-6)


def test_tensor_backward():
    rand_a = np.random.randn(2, 2)
    rand_b = np.random.randn(2, 2)
    a = Tensor(rand_a)
    b = Tensor(rand_b)
    c = a + b + 1
    f = 16 * c * 2
    d = f ** 2
    e = d.relu()
    e.backward()

    # Create PyTorch tensors
    a_torch = torch.tensor(rand_a, requires_grad=True)
    b_torch = torch.tensor(rand_b, requires_grad=True)
    c_torch = a_torch + b_torch + 1
    f_torch = 16 * c_torch * 2
    d_torch = f_torch**2
    e_torch = d_torch.relu()
    e_torch.sum().backward()

    # Compare gradients
    np.testing.assert_allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(b.grad, b_torch.grad.numpy(), atol=1e-6)
