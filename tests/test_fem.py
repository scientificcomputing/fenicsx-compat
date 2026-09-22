import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest

from fenicsx_compat.fem import interpolation_points, real_functionspace


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
