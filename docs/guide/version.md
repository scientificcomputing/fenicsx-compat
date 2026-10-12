# Version checks and optional modules

Branch on behaviour rather than on version numbers where you can. The other functions in this guide do that for you. These three are for the remaining cases. They behave the same on every DOLFINx version.

(fn-at-least)=
## `at_least`

```python
fenicsx_compat.at_least(version: str) -> bool
```

Whether the installed DOLFINx is at least `version`, compared as `packaging` versions.

(fn-before)=
## `before`

```python
fenicsx_compat.before(version: str) -> bool
```

Whether the installed DOLFINx is strictly older than `version`. Always the opposite of `at_least`.

```python
if fenicsx_compat.before("0.11.0.dev0"):
    ...  # 0.10, where facet dofs still have to be permuted by hand
```

Development builds sort before their release: `0.11.0.dev0 < 0.11.0`. To include the nightly builds that led up to a release, compare against `X.Y.0.dev0`, not `X.Y.0`.

Use a version check only when the behaviour changed but the API did not, so there is nothing to detect. For example, `basix.ufl.real_element` exists on 0.10, but a function space cannot be built from it until 0.11; see [`real_functionspace`](fn-real-functionspace).

(fn-require-module)=
## `require_module`

```python
fenicsx_compat.require_module(name: str, *, extra: str | None = None) -> types.ModuleType
```

Imports and returns the module `name`. If it is missing, raises an `ImportError` that says what to install: `pip install <extra>`, or `pip install <name>` without `extra`.

```python
gmsh = fenicsx_compat.require_module("gmsh")
h5py = fenicsx_compat.require_module("h5py", extra="scifem[h5py]")
```
