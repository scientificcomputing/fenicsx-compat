import dolfinx.fem
import dolfinx.mesh
import ufl as ufl_pkg

from fenicsx_compat.ufl import domain_of


def test_domain_of_returns_the_single_domain(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    result = domain_of(u)
    assert result is not None


def test_domain_of_raises_on_expression_with_no_domain():
    import pytest

    # A bare constant is defined on zero domains: extract_domains(...) == ().
    with pytest.raises(ValueError, match="Expected exactly one domain"):
        domain_of(ufl_pkg.as_ufl(1.0))
