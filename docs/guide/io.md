# Output and visualisation

These two functions depend on the version of ADIOS2 and PyVista, not on DOLFINx. For reading gmsh models, see [`model_to_mesh`](fn-model-to-mesh).

(fn-resolve-adios-scope)=
## `resolve_adios_scope`

```python
fenicsx_compat.resolve_adios_scope(adios2) -> module
```

Returns the namespace that holds ADIOS2's low-level Python bindings (`ADIOS`, `Mode`, ...). Raises `ImportError` if ADIOS2 was built without MPI.

| ADIOS2 | Low-level bindings |
|---|---|
| 2.9 | `adios2` |
| 2.10 and newer | `adios2.bindings`. The top-level `adios2` is the new high-level API. |

```python
adios2 = fenicsx_compat.resolve_adios_scope(fenicsx_compat.require_module("adios2"))
adios = adios2.ADIOS(MPI.COMM_WORLD)
```

The MPI check uses `is_built_with_mpi`, which ADIOS2 2.9 does not have, so in practice it needs ADIOS2 2.10 or newer.

(fn-pyvista-allow-snake-case)=
## `pyvista_allow_snake_case`

```python
fenicsx_compat.pyvista_allow_snake_case(pyvista) -> None
```

Lets PyVista objects use the snake_case names of VTK methods, which PyVista otherwise refuses.

| PyVista | Where the setting lives |
|---|---|
| 0.45, 0.46 | `pyvista._VTK_SNAKE_CASE_STATE` |
| 0.47 and newer | `pyvista.core.vtk_snake_case._state` |

```python
pyvista = fenicsx_compat.require_module("pyvista")
fenicsx_compat.pyvista_allow_snake_case(pyvista)
```

PyVista 0.45 and newer also have a public call that does the same on every version, `pyvista.vtk_snake_case("allow")`. PyVista 0.44 and older have no such guard, and this function raises `AttributeError` there.
