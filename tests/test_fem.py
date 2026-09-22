from mpi4py import MPI

import basix.ufl
import dolfinx.cpp.fem
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl

from fenicsx_compat._dolfinx_version import before
from fenicsx_compat.fem import (
    expression_eval,
    finite_element_ctor_kwargs,
    function_space_ctor_kwargs,
    interpolate,
    interpolate_to_submesh_entity_maps,
    interpolation_points,
    permute_facet_quadrature,
    permute_interpolation_data,
    real_functionspace,
)


def test_interpolation_points_returns_array(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    points = interpolation_points(V)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2


# basix.ufl.real_element is present across every supported dolfinx generation
# (confirmed by the v0.10.0 CI leg's traceback: the function existed there and
# failed inside real_functionspace on a dtype-keyword mismatch, not on absence
# of the function). So the real-element path is exercised unconditionally.


def test_real_functionspace_scalar(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = real_functionspace(msh)
    assert V.dofmap.index_map.size_global == 1


def test_real_functionspace_vector_valued(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = real_functionspace(msh, value_shape=(2,))
    assert V.dofmap.index_map.size_global * V.dofmap.index_map_bs == 2


def test_finite_element_ctor_kwargs_scalar(comm):
    ufl_el = basix.ufl.element("Lagrange", "triangle", 0, discontinuous=True)
    cpp_el = finite_element_ctor_kwargs(
        dolfinx.cpp.fem.FiniteElement_float64,
        ufl_el.basix_element._e,
        value_shape=(),
        block_size=1,
    )
    assert cpp_el is not None


def test_function_space_ctor_kwargs_builds_a_space(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    ufl_el = basix.ufl.element("Lagrange", "triangle", 0, discontinuous=True)
    cpp_el = finite_element_ctor_kwargs(
        dolfinx.cpp.fem.FiniteElement_float64,
        ufl_el.basix_element._e,
        value_shape=(),
        block_size=1,
    )
    cpp_dofmap = dolfinx.cpp.fem.create_dofmap(msh.comm, msh.topology._cpp_object, cpp_el)
    cpp_space = function_space_ctor_kwargs(
        dolfinx.cpp.fem.FunctionSpace_float64,
        msh._cpp_object,
        cpp_el,
        cpp_dofmap,
        value_shape=(),
    )
    assert cpp_space is not None


def test_interpolate_sets_function_values(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    # interpolate_f/interpolate take f shaped (value_size, num_cells * points_per_cell):
    # one column per interpolation point, per cell in `cells`, in that order.
    num_points_per_cell = interpolation_points(V).shape[0]
    values = np.full((1, num_cells * num_points_per_cell), 3.0)
    interpolate(u._cpp_object, values, cells)
    u.x.scatter_forward()
    assert np.allclose(u.x.array, 3.0)


def test_expression_eval_matches_direct_eval_call(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    n = ufl.FacetNormal(msh)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facet_indices = dolfinx.mesh.locate_entities_boundary(
        msh, tdim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    msh.topology.create_connectivity(tdim, tdim - 1)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    # On multiple ranks, a rank may own no facets on the x=0 boundary at all
    # (controller ruling F4, revised). The skip decision must be COLLECTIVE:
    # dolfinx.fem.Expression construction below calls jit.ffcx_jit(comm, ...),
    # which is collective on the mesh's communicator, so a per-rank skip would
    # let ranks with no boundary facets bail out early while the other ranks
    # block forever inside the collective JIT call, deadlocking the run.
    # Using MPI.LAND (not LOR) ensures every rank takes the same branch: if
    # any rank lacks a facet, all ranks skip together.
    if not comm.allreduce(len(facet_indices) > 0, op=MPI.LAND):
        pytest.skip("not every rank owns an x=0 boundary facet")
    facet = facet_indices[0]
    cell = f_to_c.links(facet)[0]
    local_facet = np.nonzero(c_to_f.links(cell) == facet)[0][0]

    # Points must be on the *facet* reference element (dimension tdim - 1), not
    # the cell's: with tdim-wide points, ffcx resolves entity_type="cell" and
    # rejects FacetNormal (RuntimeError) before expression_eval ever runs.
    # See scifem/src/scifem/bcs.py, which pulls interpolation points back onto
    # the facet reference element for exactly this reason. Deviation from the
    # brief's `np.zeros((1, tdim))`, which errors unconditionally.
    points = np.zeros((1, tdim - 1))
    expr = dolfinx.fem.Expression(n, points)
    entity = np.array([[cell, local_facet]], dtype=np.int32)

    result = expression_eval(expr, msh, entity)
    assert result.shape[0] == 1


# These assertions hold on both sides of the dolfinx PR #4140 boundary:
# pre-0.11 the tables are identity copies, post-0.11 they are real
# permutations, but the count and shape are the same either way. The CI
# matrix, not a patched version number, is what runs both branches.
#
# Beyond count-and-shape, we also assert on the actual values: from 0.11 the
# returned tables must be pairwise distinct, which catches a regression that
# makes two table entries collide (e.g. a formula that maps two different
# (reflection, rotation) combinations onto the same permutation). Pre-0.11,
# dolfinx applies the permutation internally, so the tables here are identity
# copies of `points` instead - asserting *that* is what the pre-0.11 leg
# guarantees. Distinctness would be false there (all rows equal `points`), so
# it must be gated on the installed dolfinx side, not asserted unconditionally.
#
# KNOWN GAP: this only checks the SET of returned tables, not their ORDER.
# Confirmed by direct experiment: patching the triangle branch's
# `rot_inv = (3 - rot) % 3 if ref == 0 else rot` to the broken `rot_inv = rot`
# still passes every assertion here, because both formulas are bijections
# from the same six (rot, ref) loop iterations onto the same six
# (reflection, rotation) argument pairs - the broken formula just pairs them
# up differently, so the six output tables come back as the same set in a
# different order. The `rot_inv` arithmetic is exactly what decides which
# permutation lands at which index (the index dolfinx looks up via
# `cell_permutation_info`), so a reordering bug here is real and silently
# uncaught by this suite. Catching it would require a position-sensitive
# oracle - e.g. re-deriving dolfinx/ffcx's cell-permutation-index convention
# independently in the test - which would just restate the implementation
# and pass for any consistent-but-wrong pairing; not attempted. An end-to-end
# facet-integration test (not a table-shape one) would be the faithful way to
# close this gap.
def _assert_facet_quadrature_permutations(perms, points):
    if before("0.11.0.dev0"):
        for p in perms:
            assert np.array_equal(p, points)
    else:
        for i in range(len(perms)):
            for j in range(i + 1, len(perms)):
                assert not np.array_equal(perms[i], perms[j]), f"perms {i} and {j} are identical"


def test_permute_facet_quadrature_interval_returns_two_permutations():
    # 0.25 != 0.75, and interval reflection maps x -> 1 - x, so the two rows
    # are not symmetric under the reflection: this is not an accidental tie.
    points = np.array([[0.25], [0.75]])
    perms = permute_facet_quadrature(basix.CellType.interval, points)
    assert len(perms) == 2
    for p in perms:
        assert p.shape == points.shape
    _assert_facet_quadrature_permutations(perms, points)


def test_permute_facet_quadrature_triangle_returns_six_permutations():
    # Coordinate-asymmetric points: neither row is a fixed point of any
    # triangle-facet reflection/rotation (e.g. a point like [0.3, 0.7] whose
    # coordinates are complementary would collapse distinct permutations onto
    # each other - that is an artifact of the test point, not of the code).
    points = np.array([[0.1, 0.9], [0.2, 0.6]])
    perms = permute_facet_quadrature(basix.CellType.triangle, points)
    assert len(perms) == 6
    for p in perms:
        assert p.shape == points.shape
    _assert_facet_quadrature_permutations(perms, points)


def test_permute_facet_quadrature_quadrilateral_returns_eight_permutations():
    # Same asymmetric points as the triangle case, reused here since they are
    # equally non-degenerate under quadrilateral-facet reflections/rotations.
    points = np.array([[0.1, 0.9], [0.2, 0.6]])
    perms = permute_facet_quadrature(basix.CellType.quadrilateral, points)
    assert len(perms) == 8
    for p in perms:
        assert p.shape == points.shape
    _assert_facet_quadrature_permutations(perms, points)


def test_permute_facet_quadrature_rejects_unsupported_cell_type():
    points = np.array([[0.25], [0.75]])
    with pytest.raises(ValueError, match="Unsupported"):
        permute_facet_quadrature(basix.CellType.tetrahedron, points)


def test_permute_interpolation_data_permutes_within_each_row(comm):
    # Pre-0.11 this reorders each row; from 0.11 it is a no-op. Either way
    # each row must still hold the same multiset of values, which is what
    # this asserts, so the same test is meaningful on every CI leg.
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    msh.topology.create_entity_permutations()
    cell_info = msh.topology.get_cell_permutation_info()

    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    original_rows = [sorted(row) for row in data]
    permute_interpolation_data(
        data,
        integration_entities=np.array([[0, 0], [1, 0]], dtype=np.int32),
        cell_permutation_info=cell_info,
        basix_element=V.element.basix_element,
        facet_type=basix.CellType.interval,
    )
    assert data.shape == (2, 3)
    assert [sorted(row) for row in data] == original_rows


def test_interpolate_to_submesh_entity_maps_builds_expression(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    # Ruling F3: route through the package's own interpolation_points() helper
    # rather than the raw `V.element.interpolation_points` attribute, which is
    # a method (not a property) on the dolfinx v0.10.0 CI leg.
    points = interpolation_points(V)
    expr = interpolate_to_submesh_entity_maps(u, points)
    assert isinstance(expr, dolfinx.fem.Expression)
