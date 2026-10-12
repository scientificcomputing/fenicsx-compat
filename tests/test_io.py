from mpi4py import MPI

import dolfinx
import dolfinx.graph
import dolfinx.mesh
import pytest

from fenicsx_compat.io import model_to_mesh, pyvista_allow_snake_case, resolve_adios_scope
from fenicsx_compat.mesh import create_cell_partitioner


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


@pytest.fixture
def unit_square_model():
    """A meshed unit square in gmsh, the model built on rank 0 only."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("unit square")
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.mesh.generate(2)
    try:
        yield gmsh.model
    finally:
        gmsh.finalize()


@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
)
def test_model_to_mesh_applies_the_ghost_mode(comm, unit_square_model, ghost_mode):
    if comm.size < 2:
        pytest.skip("ghost cells only exist with at least 2 ranks")
    mesh_data = model_to_mesh(unit_square_model, comm, 0, gdim=2, ghost_mode=ghost_mode)
    cell_map = mesh_data.mesh.topology.index_map(2)
    num_cells = comm.allreduce(cell_map.size_local, op=MPI.SUM)
    num_ghosts = comm.allreduce(cell_map.num_ghosts, op=MPI.SUM)
    assert num_cells == cell_map.size_global > 0
    if ghost_mode == dolfinx.mesh.GhostMode.none:
        assert num_ghosts == 0
    else:
        assert num_ghosts > 0


def test_model_to_mesh_respects_explicit_partitioner(comm, unit_square_model):
    calls = []
    # The partitioner dolfinx expects differs as in create_mesh: a cell partitioner on
    # 0.10 and 0.11, the graph partitioner from dolfinx PR #4403 on.
    if hasattr(dolfinx.mesh, "create_cell_partitioner"):
        base = create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    else:
        base = dolfinx.graph.partitioner()

    def spy_partitioner(*args, **kwargs):
        calls.append(1)
        return base(*args, **kwargs)

    mesh_data = model_to_mesh(unit_square_model, comm, 0, gdim=2, partitioner=spy_partitioner)
    assert mesh_data.mesh.topology.index_map(2).size_global > 0
    if comm.size > 1:
        assert calls, "explicit partitioner must be used, not silently replaced"
