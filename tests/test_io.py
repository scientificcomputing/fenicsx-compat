import pytest

from fenicsx_compat.io import import_gmshio, pyvista_allow_snake_case, resolve_adios_scope


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


def test_pyvista_allow_snake_case_sets_new_style_attribute():
    class FakePyvista:
        _VTK_SNAKE_CASE_STATE = "forbid"

    fake = FakePyvista()
    pyvista_allow_snake_case(fake)
    assert fake._VTK_SNAKE_CASE_STATE == "allow"


def test_pyvista_allow_snake_case_falls_back_to_old_style_attribute():
    class FakeState:
        _state = "forbid"

    class FakeVtkSnakeCase:
        _state = FakeState()

    class FakeCore:
        vtk_snake_case = FakeVtkSnakeCase()

    class FakePyvistaOld:
        core = FakeCore()

    fake = FakePyvistaOld()
    pyvista_allow_snake_case(fake)
    assert fake.core.vtk_snake_case._state == "allow"
