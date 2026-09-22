import dolfinx.mesh
import numpy as np

from fenicsx_compat.geometry import determine_point_ownership


def test_determine_point_ownership_finds_owning_cell(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    points = np.array([[0.5, 0.5, 0.0]])
    result = determine_point_ownership(msh, points, tol=1e-8)
    assert hasattr(result, "dest_points")
    assert hasattr(result, "dest_cells")
    assert hasattr(result, "src_owner")
