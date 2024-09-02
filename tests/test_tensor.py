import numpy as np
import torch
from cudagrad.engine import Tensor


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


def test_tensor_backward__add__tensor():
    a = Tensor(np.ones((2, 2)))
    b = Tensor(np.ones((2, 2)))
    c = a + b
    c.backward()

    assert a.grad.tolist() == [[1, 1], [1, 1]]
    assert b.grad.tolist() == [[1, 1], [1, 1]]
