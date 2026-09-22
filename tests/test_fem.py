import dolfinx.fem
import dolfinx.mesh
import numpy as np

from fenicsx_compat.fem import interpolation_points


def test_interpolation_points_returns_array(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    points = interpolation_points(V)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
