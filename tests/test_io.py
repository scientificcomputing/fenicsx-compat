import pytest

from fenicsx_compat.io import import_gmshio, resolve_adios_scope


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


def test_import_gmshio_returns_a_module():
    import types

    result = import_gmshio()
    assert isinstance(result, types.ModuleType)
    assert hasattr(result, "read_from_msh") or hasattr(result, "model_to_mesh")
