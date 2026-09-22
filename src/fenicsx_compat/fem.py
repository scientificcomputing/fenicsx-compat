import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt


def interpolation_points(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.floating]:
    """Get the interpolation points for a function space, across the method-vs-property rename."""
    try:
        return V.element.interpolation_points()
    except TypeError:
        return V.element.interpolation_points


def real_functionspace(
    mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
) -> dolfinx.fem.FunctionSpace:
    """Create a real (single-constant-per-domain) function space.

    `basix.ufl.real_element` was added in dolfinx 0.11. On dolfinx 0.10,
    no pure-Python fallback exists — scifem's pre-0.11 fallback calls its
    compiled `_scifem.create_real_functionspace_float{32,64}` extension
    (fenicsx-compat ships no compiled extensions, see spec §2), so this
    raises NotImplementedError instead.
    """
    if not hasattr(basix.ufl, "real_element"):
        raise NotImplementedError(
            "real_functionspace requires dolfinx>=0.11 (basix.ufl.real_element). "
            "On dolfinx 0.10, use scifem.create_real_functionspace() instead, "
            "which provides a compiled fallback."
        )
    el = basix.ufl.real_element(
        mesh.basix_cell(), value_shape=value_shape, dtype=mesh.geometry.x.dtype
    )
    return dolfinx.fem.functionspace(mesh, el)
