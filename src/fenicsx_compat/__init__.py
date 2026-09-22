from importlib.metadata import version as _version

from ._dolfinx_version import at_least, before
from .fem import (
    expression_eval,
    finite_element_ctor_kwargs,
    function_space_ctor_kwargs,
    interpolate,
    interpolate_to_submesh_entity_maps,
    interpolation_points,
    permute_facet_quadrature,
    permute_interpolation_data,
    real_functionspace,
)
from .geometry import determine_point_ownership
from .io import import_gmshio, pyvista_allow_snake_case, resolve_adios_scope
from .la import create_index_map, unwrap_index_map, vector
from .mesh import (
    cmap,
    create_cell_partitioner,
    create_mesh,
    dofmap,
    form_map,
    num_entity_closure_dofs,
    reconstruct_mesh,
    transfer_meshtags_to_submesh,
)
from .optional import require_module
from .petsc import (
    apply_lifting_and_set_bc,
    bcs_by_block,
    ghost_update,
    pack_coefficients,
    pack_constants,
    set_bc,
    zero_petsc_vector,
)
from .ufl import domain_of

__version__ = _version("fenicsx-compat")

__all__ = [
    "__version__",
    "at_least",
    "before",
    "require_module",
    "cmap",
    "dofmap",
    "form_map",
    "num_entity_closure_dofs",
    "create_cell_partitioner",
    "create_mesh",
    "reconstruct_mesh",
    "transfer_meshtags_to_submesh",
    "create_index_map",
    "unwrap_index_map",
    "vector",
    "interpolation_points",
    "real_functionspace",
    "finite_element_ctor_kwargs",
    "function_space_ctor_kwargs",
    "interpolate",
    "expression_eval",
    "permute_facet_quadrature",
    "permute_interpolation_data",
    "interpolate_to_submesh_entity_maps",
    "pack_constants",
    "pack_coefficients",
    "bcs_by_block",
    "zero_petsc_vector",
    "ghost_update",
    "set_bc",
    "apply_lifting_and_set_bc",
    "determine_point_ownership",
    "domain_of",
    "resolve_adios_scope",
    "import_gmshio",
    "pyvista_allow_snake_case",
]
