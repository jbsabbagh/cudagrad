# cudagrad

A minimal implementation of backpropagation for a neural network in Python, with a focus on clarity and understanding.

This project is heavily inspired by the work of [*Andrej Karpathy*](https://cs.stanford.edu/people/karpathy/) and repo [micrograd](https://github.com/karpathy/micrograd).

It rewrites the `Value` class from micrograd to use CUDA for GPU acceleration. Just a fun way to see if I could write CUDA kernels using numba.

## Caveats
The CUDA kernels are not optimized and are not meant to be. This is a toy implementation for learning purposes.

Numba's API for shared memory and threads is a bit clunky and will sometimes break when using the CUDA Numba Simulator and not the real GPU.

I didn't want to spend too much time on this as I want to explore replacing them with Triton Kernels eventually.

## Usage

```python
from cudagrad.engine import Tensor

rand_a = np.random.randn(2, 2)
rand_b = np.random.randn(2, 2)
a = Tensor(rand_a)
b = Tensor(rand_b)

# A random expression
c = a + b + 1
f = 16 * c * 2
d = f ** 2
e = d.relu()
e.backward()
```
