import numpy as np
from cudagrad.kernels.add import add_2d_numba
from cudagrad.kernels.power import power_2d_numba
from cudagrad.kernels.gradients import power_gradient_numba
from cudagrad.kernels.multiply import mul_2d_numba
from cudagrad.kernels.relu import relu_2d_numba
from cudagrad.kernels.gradients import relu_gradient_numba


class Tensor:
    def __init__(self, data: np.array, _children=(), _op="", label=""):
        self.data = data
        self.grad = np.zeros_like(data)
        self.label = label
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full_like(self.data, other))
        out = Tensor(add_2d_numba(self.data, other.data), (self, other), "+")

        def _backward():
            self.grad = add_2d_numba(self.grad, out.grad)
            other.grad = add_2d_numba(other.grad, out.grad)

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full_like(self.data, other))
        out = Tensor(mul_2d_numba(self.data, other.data), (self, other), "*")

        def _backward():
            self.grad = add_2d_numba(self.grad, mul_2d_numba(other.data, out.grad))
            other.grad = add_2d_numba(other.grad, mul_2d_numba(self.data, out.grad))

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Tensor(power_2d_numba(self.data, other), (self,), f"**{other}")

        def _backward():
            new_grad = power_gradient_numba(self.data, other)
            self.grad = add_2d_numba(self.grad, new_grad)

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(relu_2d_numba(self.data), (self,), "ReLU")

        def _backward():
            new_grad = relu_gradient_numba(self.data, out.grad)
            self.grad = add_2d_numba(self.grad, new_grad)

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, Tensor={self.grad})"
