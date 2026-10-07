from importlib.metadata import version as _version

from ._dolfinx_version import at_least, before
from .fem import (
    expression_eval,
    finite_element_ctor_kwargs,
    interpolate,
    interpolate_to_submesh_entity_maps,
    permute_facet_quadrature,
    permute_interpolation_data,
    real_functionspace,
)
from .io import pyvista_allow_snake_case, resolve_adios_scope
from .la import create_index_map, index_to_dest_ranks, unwrap_index_map, vector
from .mesh import (
    cell_permutation_info,
    cmap,
    create_cell_partitioner,
    create_mesh,
    dofmap,
    facet_permutations,
    form_map,
    reconstruct_mesh,
    transfer_meshtags_to_submesh,
)
from .optional import require_module
from .petsc import (
    apply_lifting_and_set_bc,
    bcs_by_block,
    ghost_update,
    set_bc,
    zero_petsc_vector,
)
from .ufl import apply_pullback_inverse, domain_of

__version__ = _version("fenicsx-compat")

__all__ = [
    "__version__",
    "at_least",
    "before",
    "require_module",
    "cmap",
    "dofmap",
    "form_map",
    "cell_permutation_info",
    "facet_permutations",
    "create_cell_partitioner",
    "create_mesh",
    "reconstruct_mesh",
    "transfer_meshtags_to_submesh",
    "create_index_map",
    "unwrap_index_map",
    "index_to_dest_ranks",
    "vector",
    "real_functionspace",
    "finite_element_ctor_kwargs",
    "interpolate",
    "expression_eval",
    "permute_facet_quadrature",
    "permute_interpolation_data",
    "interpolate_to_submesh_entity_maps",
    "bcs_by_block",
    "zero_petsc_vector",
    "ghost_update",
    "set_bc",
    "apply_lifting_and_set_bc",
    "domain_of",
    "apply_pullback_inverse",
    "resolve_adios_scope",
    "pyvista_allow_snake_case",
]
