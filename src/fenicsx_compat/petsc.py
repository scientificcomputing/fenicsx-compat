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

    import dolfinx.fem.petsc

    def set_bc(b: PETSc.Vec, bcs, x0=None, alpha: float = 1.0) -> None:
        """Set boundary condition values in a vector, across the nested-vs-single-form
        `set_bc` API."""
        try:
            dolfinx.fem.petsc.set_bc(b, bcs, x0=x0, alpha=alpha)
        except AttributeError:
            for _bcs in bcs:
                dolfinx.fem.petsc.set_bc(b, _bcs, x0=x0, alpha=alpha)
        except TypeError:
            # bcs shouldn't be a nested list here (e.g. for a single form).
            assert len(bcs) == 1, "bcs should be a single DirichletBC or a list of DirichletBCs."
            dolfinx.fem.petsc.set_bc(b, bcs[0], x0=x0, alpha=alpha)

    def apply_lifting_and_set_bc(
        b: PETSc.Vec,
        a,
        bcs,
        x: PETSc.Vec | None = None,
        alpha: float = 1.0,
    ) -> None:
        r"""Apply lifting to a vector and set boundary conditions.

        Convenience function to apply lifting and set boundary conditions
        for single, blocked, or nested forms. Modifies `b` such that::

            b = [b_free - alpha * sum_i(a[i] @ (u_bc[i] - x[i])), u_bc[0], ..., u_bc[n]]

        Args:
            b: The vector to apply lifting to.
            a: Form or nested sequence of forms to apply lifting from.
            bcs: The boundary conditions to apply.
            x: Vector to subtract from the boundary conditions (e.g. in a Newton iteration).
            alpha: Scaling factor for the boundary conditions.
        """
        # Ruling F6 (revised): use this module's own bcs_by_block (Task 21)
        # instead of scifem's `dolfinx.fem.bcs_by_block`, which on failure
        # hands every bc to every space unfiltered -- silently wrong for a
        # blocked/nested system. That much of scifem's try/except was a real
        # bug and stays deleted.
        #
        # But `dolfinx.fem.extract_function_spaces` also refuses an explicit
        # index for a bare single form (`ValueError: index must be None for
        # a single form`, confirmed), and for that case `bcs0 = bcs1 = bcs`
        # is not a workaround, it is the correct answer: a single form has
        # exactly one block, so every bc in `bcs` belongs to it. The
        # docstring above promises "single, blocked, or nested forms", so
        # single-form support is part of this function's contract and must
        # stay -- but selected by an explicit, positive check on `a`'s shape
        # (never by catching whatever exception extract_function_spaces
        # happens to raise, which could silently swallow an unrelated bug).
        if isinstance(a, (list, tuple)):
            # Blocked/nested: `a` is a 2D array of forms, one row per block.
            bcs0 = bcs_by_block(dolfinx.fem.extract_function_spaces(a, 0), bcs)
            bcs1 = bcs_by_block(dolfinx.fem.extract_function_spaces(a, 1), bcs)
            a_lifting = a
        else:
            # Single form: one block, so every bc applies to it. `set_bc`
            # (below) accepts `bcs` in its bare/flat shape directly, but
            # `dolfinx.fem.petsc.apply_lifting` always wants `a` and `bcs` one
            # level more nested than that for a plain (non-blocked, non-nest)
            # vector -- confirmed: a bare `a` raises `TypeError: 'Form'
            # object is not iterable`, and a flat `bcs` raises `TypeError:
            # 'DirichletBC' object is not iterable` inside apply_lifting.
            bcs0 = bcs
            bcs1 = [bcs]
            a_lifting = [a]

        dolfinx.fem.petsc.apply_lifting(b, a_lifting, bcs=bcs1, x0=x, alpha=alpha)
        ghost_update(b, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs0, x0=x, alpha=alpha)
        ghost_update(b, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

else:

    def zero_petsc_vector(b) -> None:
        raise RuntimeError("petsc4py is not available. Cannot zero vector.")

    def ghost_update(x, insert_mode, scatter_mode) -> None:
        raise RuntimeError("petsc4py is not available. Cannot ghost update vector.")

    def set_bc(b, bcs, x0=None, alpha: float = 1.0) -> None:
        raise RuntimeError("petsc4py is not available. Cannot set boundary conditions.")

    def apply_lifting_and_set_bc(b, a, bcs, x=None, alpha: float = 1.0) -> None:
        raise RuntimeError(
            "petsc4py is not available. Cannot apply lifting and set boundary conditions."
        )


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
