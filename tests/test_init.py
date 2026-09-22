import fenicsx_compat


def test_all_public_functions_are_re_exported_at_top_level():
    expected = [
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
    for name in expected:
        assert hasattr(fenicsx_compat, name), f"fenicsx_compat.{name} is not re-exported"
