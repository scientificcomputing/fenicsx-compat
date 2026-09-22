from mpi4py import MPI

import pytest


@pytest.fixture
def comm():
    return MPI.COMM_WORLD
