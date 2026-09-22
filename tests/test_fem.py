import basix.ufl
import dolfinx.cpp.fem
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl

from fenicsx_compat.fem import (
    expression_eval,
    finite_element_ctor_kwargs,
    function_space_ctor_kwargs,
    interpolate,
    interpolation_points,
    real_functionspace,
)


def test_interpolation_points_returns_array(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    points = interpolation_points(V)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2


# basix.ufl.real_element arrived with dolfinx 0.11; on the v0.10.0 CI leg
# there is no pure-Python fallback, so the compat function says so instead.
# Which branch runs is decided by the installed basix, not by a mock.
HAS_REAL_ELEMENT = hasattr(basix.ufl, "real_element")


def test_real_functionspace_scalar(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    if HAS_REAL_ELEMENT:
        V = real_functionspace(msh)
        assert V.dofmap.index_map.size_global == 1
    else:
        with pytest.raises(NotImplementedError, match="0.11"):
            real_functionspace(msh)


def test_real_functionspace_vector_valued(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    if HAS_REAL_ELEMENT:
        V = real_functionspace(msh, value_shape=(2,))
        assert V.dofmap.index_map.size_global * V.dofmap.index_map_bs == 2
    else:
        with pytest.raises(NotImplementedError, match="0.11"):
            real_functionspace(msh, value_shape=(2,))


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
    # (controller ruling F4): skip on that rank rather than index an empty array.
    if len(facet_indices) == 0:
        pytest.skip("no x=0 boundary facets on this rank")
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
