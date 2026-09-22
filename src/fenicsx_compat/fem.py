import dolfinx.fem
import numpy as np
import numpy.typing as npt


def interpolation_points(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.floating]:
    """Get the interpolation points for a function space, across the method-vs-property rename."""
    try:
        return V.element.interpolation_points()
    except TypeError:
        return V.element.interpolation_points
