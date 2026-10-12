# fenicsx-compat
```{include} README.md
:start-after: "# fenicsx-compat"
:end-before: "# Installation"
```

## Supported versions

| | Supported |
|---|---|
| DOLFINx | 0.10, 0.11 and `main` (the nightly builds, currently 0.12.0.dev0) |
| Python | 3.12 and newer |

Throughout this guide, **main** means the current development branch of DOLFINx, which will become 0.12.

## How to use this guide

- {doc}`docs/choosing` maps the cross-version problem you have to the function that solves it. Start there.
- The **function guide** has one entry per function. Each entry says what the function does, how DOLFINx 0.10, 0.11 and main do the same thing natively, and how to call the function so that one piece of code runs on all three.
- {doc}`docs/native` lists the things that look version-dependent but are not: from 0.10 on, plain DOLFINx calls work on every supported version, and there is no wrapper for them.
- The {doc}`docs/api` has the full docstrings.

Every function is importable from the top-level package:

```python
import fenicsx_compat

cmap = fenicsx_compat.cmap(mesh)
```

The examples in the guide are short, and assume

```python
from mpi4py import MPI
import dolfinx
import numpy as np
import fenicsx_compat
```

and, where they use one, an existing `mesh`, function space `V` or form.
