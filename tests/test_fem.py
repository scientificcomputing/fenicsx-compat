import basix.ufl
import dolfinx.cpp.fem
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest

from fenicsx_compat.fem import (
    finite_element_ctor_kwargs,
    function_space_ctor_kwargs,
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
