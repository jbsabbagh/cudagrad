import os
import pytest


@pytest.fixture(scope="module")
def init_cuda_simulator():
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

    yield

    os.environ.pop("NUMBA_ENABLE_CUDASIM", None)
