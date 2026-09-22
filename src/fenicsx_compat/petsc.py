from collections.abc import Sequence

import dolfinx.cpp.fem
import dolfinx.fem
import numpy.typing as npt


def pack_constants(form: dolfinx.fem.Form) -> npt.NDArray:
    """Pack a form's constants, across the public-API-vs-cpp-object split."""
    try:
        return dolfinx.fem.pack_constants(form)
    except AttributeError:
        return dolfinx.cpp.fem.pack_constants(form._cpp_object)


def pack_coefficients(form: dolfinx.fem.Form) -> dict:
    """Pack a form's coefficients, across the public-API-vs-cpp-object split."""
    try:
        return dolfinx.fem.pack_coefficients(form)
    except AttributeError:
        return dolfinx.cpp.fem.pack_coefficients(form._cpp_object)


def bcs_by_block(
    spaces: Sequence[dolfinx.fem.FunctionSpace | None],
    bcs: Sequence[dolfinx.fem.DirichletBC],
) -> list[list[dolfinx.fem.DirichletBC]]:
    """Arrange boundary conditions by the space they constrain, across dolfinx 0.11/0.12.

    `dolfinx.fem.bcs_by_block` cannot be called directly because 0.11 and
    0.12 disagree about which side of the containment check is a Python
    wrapper and which is a cpp object, in opposite directions (dolfinx PR
    #4342). Normalizing both sides to cpp objects and calling the cpp
    `contains` directly is stable across both and accepts either flavour
    of space from the caller. (dolfinx PR #4312)
    """

    def _cpp(space):
        return getattr(space, "_cpp_object", space)

    return [
        [bc for bc in bcs if _cpp(V).contains(_cpp(bc.function_space))] if V is not None else []
        for V in spaces
    ]
