import numpy as np
import numpy.typing as npt

import dolfinx


def cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the coordinate element for a mesh's geometry.

    Ported from scifem/io4dolfinx's `compat.cmap`, across three
    generations of `mesh.geometry`'s cmap API: plural `cmaps[]`, callable
    `cmap()`, and attribute `cmap`.
    """
    if hasattr(mesh.geometry, "cmaps"):
        return mesh.geometry.cmaps[0]
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    return mesh.geometry.cmap


def dofmap(mesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.int32]:
    """Get the geometry dofmap for a mesh, across the same generations as `cmap`."""
    if hasattr(mesh.geometry, "dofmaps"):
        return mesh.geometry.dofmaps[0]
    if callable(mesh.geometry.dofmap):
        return mesh.geometry.dofmap()
    return mesh.geometry.dofmap


def form_map(form: dolfinx.fem.Form) -> tuple[dolfinx.common.IndexMap, int]:
    """Get the index map and block size for a linear form's test space.

    `FunctionSpace.dofmaps` was a callable method in older dolfinx and is a
    subscriptable sequence in newer dolfinx.
    """
    try:
        return (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
    except TypeError:
        return (
            form.function_spaces[0].dofmaps[0].index_map,
            form.function_spaces[0].dofmaps[0].index_map_bs,
        )


def num_entity_closure_dofs(dof_layout: dolfinx.cpp.fem.ElementDofLayout, dim: int) -> int:
    """Get the number of dofs in the closure of a `dim`-dimensional entity."""
    if hasattr(dof_layout, "num_entity_closure_dofs"):
        return dof_layout.num_entity_closure_dofs(dim)
    return len(dof_layout.entity_closure_dofs(dim, 0))
