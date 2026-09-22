import dolfinx.cpp.geometry
import dolfinx.geometry
import dolfinx.mesh
import numpy as np
import numpy.typing as npt


def determine_point_ownership(
    mesh: dolfinx.mesh.Mesh, points: npt.NDArray[np.floating], tol: float
):
    """Determine which process owns each point and which cell it lies within.

    Across the cpp-vs-Python namespace move and the `tol`-vs-`padding`
    rename (dolfinx >=0.9.0; the exact `dolfinx.__version__ == "0.8.0"`
    branch found in the source repos is below fenicsx-compat's 0.10 floor
    and is not ported).
    """
    try:
        # Pre-0.9 cpp signature (positional `tol`, cpp mesh object); on this
        # install the public dolfinx.geometry API (keyword `padding`) is the
        # right shape, so this call is expected to raise TypeError and fall
        # through to the except branch below.
        return dolfinx.cpp.geometry.determine_point_ownership(  # type: ignore[call-overload]
            mesh._cpp_object, points, tol
        )
    except TypeError:
        return dolfinx.geometry.determine_point_ownership(mesh, points, padding=tol)
