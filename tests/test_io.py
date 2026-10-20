from mpi4py import MPI

import dolfinx
import dolfinx.graph
import dolfinx.mesh
import pytest

from fenicsx_compat.io import (
    model_to_mesh,
    pyvista_allow_snake_case,
    resolve_adios_scope,
    vtk_cell_type,
)
from fenicsx_compat.mesh import create_cell_partitioner, dofmap

# gmsh is optional: the model_to_mesh tests run where it is installed and skip elsewhere.
try:
    import gmsh
except ImportError:
    gmsh = None  # type: ignore[assignment]

requires_gmsh = pytest.mark.skipif(gmsh is None, reason="gmsh is not installed")


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


@requires_gmsh
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


@requires_gmsh
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


@pytest.mark.parametrize(
    "create, vtk_id",
    [
        (lambda comm: dolfinx.mesh.create_unit_interval(comm, 4), 68),
        (lambda comm: dolfinx.mesh.create_unit_square(comm, 2, 2), 69),
        (
            lambda comm: dolfinx.mesh.create_unit_square(
                comm, 2, 2, cell_type=dolfinx.mesh.CellType.quadrilateral
            ),
            70,
        ),
        (lambda comm: dolfinx.mesh.create_unit_cube(comm, 2, 2, 2), 71),
        (
            lambda comm: dolfinx.mesh.create_unit_cube(
                comm, 2, 2, 2, cell_type=dolfinx.mesh.CellType.hexahedron
            ),
            72,
        ),
    ],
)
def test_vtk_cell_type_is_the_arbitrary_lagrange_type_of_the_mesh_cell(comm, create, vtk_id):
    msh = create(comm)
    num_nodes = dofmap(msh).shape[1]
    # Passing the topological dimension as the node count, as callers did before
    # dolfinx PR #4621, raises for quadrilaterals and hexahedra there.
    assert vtk_cell_type(msh.topology.cell_type, num_nodes) == vtk_id


@pytest.mark.parametrize(
    "cell_type, num_nodes, vtk_id",
    [
        (dolfinx.mesh.CellType.triangle, 6, 69),
        (dolfinx.mesh.CellType.quadrilateral, 9, 70),
        (dolfinx.mesh.CellType.tetrahedron, 10, 71),
        (dolfinx.mesh.CellType.hexahedron, 27, 72),
    ],
)
def test_vtk_cell_type_of_quadratic_lagrange_cells(cell_type, num_nodes, vtk_id):
    assert vtk_cell_type(cell_type, num_nodes) == vtk_id
