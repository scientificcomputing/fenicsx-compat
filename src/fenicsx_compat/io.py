def resolve_adios_scope(adios2):
    """Resolve the ADIOS2 Python module scope, across the `adios2.bindings` submodule split."""
    scope = adios2.bindings if hasattr(adios2, "bindings") else adios2
    if not scope.is_built_with_mpi:
        raise ImportError("ADIOS2 must be built with MPI support")
    return scope
