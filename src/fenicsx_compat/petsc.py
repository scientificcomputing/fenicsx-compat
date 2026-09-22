from collections.abc import Sequence

import dolfinx
import dolfinx.cpp.fem
import dolfinx.fem
import numpy.typing as npt

if dolfinx.has_petsc4py:
    from petsc4py import PETSc

    try:
        import dolfinx.la.petsc

        def zero_petsc_vector(b: PETSc.Vec) -> None:
            """Zero a PETSc vector, including ghosts."""
            dolfinx.la.petsc._zero_vector(b)

        def ghost_update(x: PETSc.Vec, insert_mode, scatter_mode) -> None:
            """Ghost-update a PETSc vector."""
            dolfinx.la.petsc._ghost_update(x, insert_mode, scatter_mode)

    except ModuleNotFoundError:

        def zero_petsc_vector(b: PETSc.Vec) -> None:
            """Zero a PETSc vector, including ghosts."""
            if b.getType() == PETSc.Vec.Type.NEST:
                for b_sub in b.getNestSubVecs():
                    with b_sub.localForm() as b_local:
                        b_local.set(0.0)
            else:
                with b.localForm() as b_loc:
                    b_loc.set(0)

        def ghost_update(x: PETSc.Vec, insert_mode, scatter_mode) -> None:
            """Ghost-update a PETSc vector."""
            if x.getType() == PETSc.Vec.Type.NEST:
                for x_sub in x.getNestSubVecs():
                    x_sub.ghostUpdate(addv=insert_mode, mode=scatter_mode)
            else:
                x.ghostUpdate(addv=insert_mode, mode=scatter_mode)

else:

    def zero_petsc_vector(b) -> None:
        raise RuntimeError("petsc4py is not available. Cannot zero vector.")

    def ghost_update(x, insert_mode, scatter_mode) -> None:
        raise RuntimeError("petsc4py is not available. Cannot ghost update vector.")


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
