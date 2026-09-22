import inspect

import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt

from fenicsx_compat._dolfinx_version import before


def interpolation_points(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.floating]:
    """Get the interpolation points for a function space, across the method-vs-property rename.

    The exact release boundary is not pinned upstream: dolfinx-adjoint
    (the source this was ported from) dispatches via `try/except
    TypeError` rather than a version check. As an observed fact on this
    install, dolfinx 0.12.0.dev0 exposes `element.interpolation_points`
    as a property, not a callable method.
    """
    try:
        # interpolation_points was a callable method pre-dolfinx-0.12; on
        # this install it is a property (already an ndarray), so calling it
        # is expected to raise TypeError and fall through to the except
        # branch below.
        return V.element.interpolation_points()  # type: ignore[operator]
    except TypeError:
        return V.element.interpolation_points


def real_functionspace(
    mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
) -> dolfinx.fem.FunctionSpace:
    """Create a real (single-constant-per-domain) function space.

    `basix.ufl.real_element` is present across this project's whole
    supported dolfinx range (confirmed on the v0.10.0 CI leg, whose
    traceback showed the function existing but rejecting a `dtype`
    keyword it did not yet accept there). What changed later is that it
    gained a `dtype` parameter; the exact release is not pinned upstream,
    so this detects it by `inspect.signature` rather than a version
    check, and only passes `dtype=` when the installed signature has
    that parameter.

    The `hasattr(basix.ufl, "real_element")` guard below is not a
    supported-version gap: `basix` is a separate package from `dolfinx`
    and can be version-skewed against it, so a clear NotImplementedError
    beats an AttributeError deeper in the call.
    """
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
    return dolfinx.fem.functionspace(mesh, el)


def finite_element_ctor_kwargs(
    constructor,
    basix_element,
    *,
    value_shape: tuple[int, ...],
    block_size: int,
    symmetric: bool = False,
):
    """Construct a cpp FiniteElement, across the block_shape/block_size kwarg rename.

    The exact release boundary is not pinned; scifem (the source this
    was ported from) and this project's design spec (§5.3) both
    dispatch on the `TypeError` raised by the `block_shape` kwarg rather
    than a version number.
    """
    try:
        return constructor(basix_element, block_shape=value_shape, symmetric=symmetric)
    except TypeError:
        return constructor(basix_element, block_size=block_size, symmetric=symmetric)


def function_space_ctor_kwargs(
    constructor,
    mesh_cpp,
    cpp_element,
    cpp_dofmap,
    *,
    value_shape: tuple[int, ...],
):
    """Construct a cpp FunctionSpace, across a `value_shape`-kwarg-existence split.

    The exact release boundary is not pinned; this dispatches on the
    `TypeError` raised by omitting `value_shape` rather than a version
    number. This split tracks the `block_shape`/`block_size` rename in
    `finite_element_ctor_kwargs` (both changes appear together in
    scifem's FiniteElement/FunctionSpace construction), and neither
    scifem nor this project's design spec (§5.3) records a version
    reference for it.
    """
    try:
        return constructor(mesh_cpp, cpp_element, cpp_dofmap)
    except TypeError:
        return constructor(mesh_cpp, cpp_element, cpp_dofmap, value_shape=value_shape)


def interpolate(cpp_function, values: npt.NDArray, cells: npt.NDArray[np.int32]) -> None:
    """Interpolate raw values into a cpp Function, across the interpolate_f/interpolate rename."""
    if hasattr(cpp_function, "interpolate_f"):
        cpp_function.interpolate_f(values, cells)
    else:
        cpp_function.interpolate(values, cells)


def expression_eval(expr, domain, entity: npt.NDArray[np.int32]) -> npt.NDArray:
    """Evaluate an Expression at integration entities.

    Across dolfinx PR #4140, the entity-array shape for `Expression.eval`
    changed from a flat 1D array to a 2D `[[cell, local_entity]]` array.
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

    Before dolfinx PR #4140 (dolfinx<0.11.0.dev0), facet-quadrature-point
    permutation was handled internally in dolfinx's C++ layer, so the
    same points apply to every permutation. From 0.11.0.dev0, FFCx
    expects the caller to precompute the permuted tables itself, via
    `ffcx.ir.elementtables` (an internal, non-public FFCx module).
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
    """Build an Expression, across the `entity_maps`-kwarg-existence split on `Expression`."""
    if "entity_maps" in inspect.signature(dolfinx.fem.Expression).parameters:
        return dolfinx.fem.Expression(volume_function, points, entity_maps=entity_maps)
    return dolfinx.fem.Expression(volume_function, points)
