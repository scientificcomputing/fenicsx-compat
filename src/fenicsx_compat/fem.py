import inspect

import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt

from ._dolfinx_version import at_least, before


def real_functionspace(
    mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
) -> dolfinx.fem.FunctionSpace:
    """Create a real (single-constant-per-domain) function space.

    This is a deliberate exception to this project's prefer-introspection
    rule (spec §4): `basix.ufl.real_element` *exists* and *constructs*
    successfully on both sides of the dolfinx 0.10/0.11 boundary, so
    `hasattr` cannot distinguish them. What actually differs is deeper:
    at dolfinx v0.10.0 the element `real_element` returns is a
    non-functional stub whose `basix_element` property raises a bare,
    message-less `NotImplementedError` (confirmed by the v0.10.0 CI
    traceback: `dolfinx.fem.functionspace` accesses `ufl_e.basix_element`
    while building the space, which is exactly where it dies). From
    dolfinx 0.11 that property works and `dolfinx.fem.functionspace`
    succeeds (confirmed locally on this install, dolfinx 0.12.0.dev0).
    Since no `hasattr`/signature probe can see into that property without
    constructing a whole function space, we gate on `at_least("0.11.0")`
    instead - a hard version check, used here only because introspection
    genuinely cannot distinguish the behaviour. Do not "fix" this back to
    an introspection-only check.

    Separately, and unrelated to the above, `real_element` gained a
    `dtype` parameter in basix 0.11 (basix PR #1001). That part *is*
    detected by `inspect.signature`, and `dtype=` is only passed when the
    installed signature has that parameter.

    The `hasattr(basix.ufl, "real_element")` guard below covers a
    different failure mode: `basix` is a separate, independently
    installable package from `dolfinx`, so it can be version-skewed
    against it. A clear NotImplementedError beats an AttributeError
    deeper in the call.
    """
    if not at_least("0.11.0"):
        raise NotImplementedError(
            "real_functionspace requires dolfinx>=0.11.0: at v0.10.0, "
            "basix.ufl.real_element exists but returns an element whose "
            "basix_element property raises NotImplementedError, so "
            "dolfinx.fem.functionspace cannot build a real space from it. "
            "Use scifem.create_real_functionspace() instead on dolfinx 0.10."
        )
    if not hasattr(basix.ufl, "real_element"):
        raise NotImplementedError(
            "real_functionspace requires basix.ufl.real_element, which is not present "
            "in the installed basix. This indicates a basix/dolfinx version skew, not "
            "an unsupported dolfinx release."
        )
    kwargs: dict = {"value_shape": value_shape}
    if "dtype" in inspect.signature(basix.ufl.real_element).parameters:
        kwargs["dtype"] = mesh.geometry.x.dtype
    el = basix.ufl.real_element(mesh.basix_cell(), **kwargs)
    try:
        return dolfinx.fem.functionspace(mesh, el)
    except NotImplementedError as e:
        raise NotImplementedError(
            "real_functionspace: dolfinx.fem.functionspace could not build a real "
            "function space from basix.ufl.real_element on this basix/dolfinx "
            "install (the underlying error was a bare, message-less "
            "NotImplementedError from basix's element stub). This is the same "
            "failure mode observed below dolfinx 0.11; if it is happening above "
            "that version, this is an unexpected basix/dolfinx pairing - consider "
            "scifem.create_real_functionspace() as a fallback."
        ) from e


def finite_element_ctor_kwargs(
    constructor,
    basix_element,
    *,
    gdim: int,
    value_shape: tuple[int, ...] | None,
    symmetric: bool = False,
):
    """Construct a single-element cpp FiniteElement, across the `gdim` addition.

    dolfinx PR #4511 (post-0.11) added the mesh's geometric dimension as
    a required constructor argument, needed for elements on manifolds.
    0.10 and 0.11 take `(element, block_shape, symmetric)` only, and
    `gdim` is ignored there. Dispatches on the `TypeError` nanobind
    raises for the unmatched overload, before any work is done.

    `constructor` is `dolfinx.cpp.fem.FiniteElement_float32` or
    `_float64`; `value_shape` is passed as dolfinx's `block_shape`
    (`None` for an unblocked element).
    """
    try:
        return constructor(basix_element, gdim=gdim, block_shape=value_shape, symmetric=symmetric)
    except TypeError:
        return constructor(basix_element, block_shape=value_shape, symmetric=symmetric)


def dirichletbc_init_kwargs(
    V: dolfinx.fem.FunctionSpace, g: dolfinx.fem.Function | dolfinx.fem.Constant
) -> dict:
    """Keyword arguments for wrapping a cpp DirichletBC in `dolfinx.fem.DirichletBC`.

    dolfinx PR #4342 (post-0.11) made the Python wrapper hold the Python
    space and value it is built with, so its initialiser takes
    `(bc, V, g)`. 0.10 and 0.11 take `(bc)` only and read both back from
    the cpp object. Unpack the result into the constructor, or into
    `super().__init__` from a subclass: `DirichletBC(cpp_bc, **kwargs)`.
    `V` is the constrained space, which need not be `g.function_space`.
    Detected from the initialiser's signature.
    """
    if "V" in inspect.signature(dolfinx.fem.DirichletBC.__init__).parameters:
        return {"V": V, "g": g}
    return {}


def interpolate(cpp_function, values: npt.NDArray, cells: npt.NDArray[np.int32]) -> None:
    """Interpolate raw values into a cpp Function, across the interpolate_f/interpolate rename."""
    if hasattr(cpp_function, "interpolate_f"):
        cpp_function.interpolate_f(values, cells)
    else:
        cpp_function.interpolate(values, cells)


def expression_eval(expr, domain, entity: npt.NDArray[np.int32]) -> npt.NDArray:
    """Evaluate an Expression at integration entities.

    Facet entities are a 2D `[[cell, local_entity]]` array from dolfinx
    PR #3625, which is in 0.10, so `expr.eval(domain, entity)` works the
    same on every supported version. The retry with the entities
    flattened is for the 1D shape of older dolfinx.
    """
    try:
        return expr.eval(domain, entity)
    except (AttributeError, AssertionError):
        return expr.eval(domain, entity.flatten())


_FACET_PERMUTATION_COUNTS = {
    basix.CellType.interval: 2,
    basix.CellType.triangle: 6,
    basix.CellType.quadrilateral: 8,
}


def permute_facet_quadrature(
    facet_type: basix.CellType, points: npt.NDArray[np.floating]
) -> list[npt.NDArray[np.floating]]:
    """Build the per-permutation facet quadrature point table.

    Entry `p` holds the points to use for a facet with permutation `p`, so
    that an Expression evaluated on a facet is evaluated at `points` as
    seen from the cell. Before dolfinx PR #4140 (dolfinx<0.11.0.dev0), facet
    Expressions are not permuted at all, so every entry is `points`. From
    0.11.0.dev0, dolfinx and FFCx (FFCx PR #836) map the points through the
    facet's global orientation, so each entry is `points` with the inverse
    of that permutation applied, computed with `ffcx.ir.elementtables` (an
    internal, non-public FFCx module).
    """
    if before("0.11.0.dev0"):
        if facet_type not in _FACET_PERMUTATION_COUNTS:
            raise ValueError(f"Unsupported facet_type={facet_type!r}")
        return [points for _ in range(_FACET_PERMUTATION_COUNTS[facet_type])]

    try:
        from ffcx.ir.elementtables import (
            permute_quadrature_interval,
            permute_quadrature_quadrilateral,
            permute_quadrature_triangle,
        )
    except ImportError as e:
        raise NotImplementedError(
            "permute_facet_quadrature requires ffcx.ir.elementtables, an internal "
            "FFCx module; it may have moved in your installed ffcx version."
        ) from e

    if facet_type == basix.CellType.interval:
        return [permute_quadrature_interval(points, ref) for ref in range(2)]
    elif facet_type == basix.CellType.triangle:
        perms = []
        for rot in range(3):
            for ref in range(2):
                rot_inv = (3 - rot) % 3 if ref == 0 else rot
                perms.append(permute_quadrature_triangle(points, ref, rot_inv))
        return perms
    elif facet_type == basix.CellType.quadrilateral:
        perms = []
        for rot in range(4):
            for ref in range(2):
                rot_inv = (4 - rot) % 4 if ref == 0 else rot
                perms.append(permute_quadrature_quadrilateral(points, ref, rot_inv))
        return perms
    else:
        raise ValueError(f"Unsupported facet_type={facet_type!r}")


def permute_interpolation_data(
    data: npt.NDArray,
    integration_entities: npt.NDArray[np.int32],
    cell_permutation_info: npt.NDArray[np.uint32],
    basix_element,
    facet_type: basix.CellType,
) -> None:
    """In-place permute interpolation data for the pre-dolfinx-PR-#4140 dof ordering.

    No-op from dolfinx>=0.11.0.dev0, where dolfinx applies this
    permutation internally.
    """
    if not before("0.11.0.dev0"):
        return
    for i in range(integration_entities.shape[0]):
        perm = np.arange(data.shape[1], dtype=np.int32)
        basix_element.permute_subentity_closure_inv(
            perm,
            cell_permutation_info[integration_entities[i, 0]],
            facet_type,
            int(integration_entities[i, 1]),
        )
        data[i] = data[i][perm]


def interpolate_to_submesh_entity_maps(
    volume_function: dolfinx.fem.Function,
    points: npt.NDArray[np.floating],
    entity_maps=None,
) -> dolfinx.fem.Expression:
    """Build an Expression, across the `entity_maps`-kwarg-existence split on `Expression`.

    `entity_maps` is passed from dolfinx PR #4140 (0.11) on. dolfinx 0.10
    has no such argument, so there it is dropped.
    """
    if "entity_maps" in inspect.signature(dolfinx.fem.Expression).parameters:
        return dolfinx.fem.Expression(volume_function, points, entity_maps=entity_maps)
    return dolfinx.fem.Expression(volume_function, points)
