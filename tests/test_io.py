import pytest

from fenicsx_compat.io import resolve_adios_scope


def test_resolve_adios_scope_returns_the_mpi_built_scope():
    adios2 = pytest.importorskip("adios2")
    scope = resolve_adios_scope(adios2)
    assert scope.is_built_with_mpi


def test_resolve_adios_scope_raises_when_not_built_with_mpi():
    class FakeScope:
        is_built_with_mpi = False

    class FakeAdios2:
        bindings = FakeScope()

    with pytest.raises(ImportError, match="MPI"):
        resolve_adios_scope(FakeAdios2())
