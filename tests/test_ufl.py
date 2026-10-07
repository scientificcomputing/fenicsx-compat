import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl as ufl_pkg

from fenicsx_compat.ufl import _INVERSE_PULLBACKS, apply_pullback_inverse, domain_of


def test_domain_of_returns_the_single_domain(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    result = domain_of(u)
    assert result is not None


def test_domain_of_raises_on_expression_with_no_domain():
    # A bare constant is defined on zero domains: extract_domains(...) == ().
    with pytest.raises(ValueError, match="Expected exactly one domain"):
        domain_of(ufl_pkg.as_ufl(1.0))


_PULLBACKS = [
    ("IdentityPullback", ufl_pkg.pullback.identity_pullback),
    ("ContravariantPiola", ufl_pkg.pullback.contravariant_piola),
    ("CovariantPiola", ufl_pkg.pullback.covariant_piola),
]


def _distorted_quad_mesh(comm):
    # Non-affine cells, so the Jacobian varies within each cell and a wrong
    # Piola formula cannot agree with the right one by accident.
    msh = dolfinx.mesh.create_unit_square(comm, 3, 3, cell_type=dolfinx.mesh.CellType.quadrilateral)
    x = msh.geometry.x
    x[:, 0] += 0.15 * x[:, 1] ** 2
    x[:, 1] += 0.1 * np.sin(3 * x[:, 0])
    return msh


def _eval_on_owned_cells(msh, expr):
    points = np.array([[0.2, 0.3], [0.7, 0.6]])
    cells = np.arange(msh.topology.index_map(2).size_local, dtype=np.int32)
    return dolfinx.fem.Expression(expr, points).eval(msh, cells)


@pytest.mark.parametrize("name, pullback", _PULLBACKS)
def test_apply_pullback_inverse_closed_forms_match_ufl(comm, name, pullback):
    # The closed forms are only used where UFL lacks apply_inverse (before
    # UFL 2026.2), so check them against UFL's own on the legs that have it.
    if not hasattr(pullback, "apply_inverse"):
        pytest.skip("this UFL has no apply_inverse to compare against")
    msh = _distorted_quad_mesh(comm)
    X = ufl_pkg.SpatialCoordinate(msh)
    f = ufl_pkg.as_vector((X[0] ** 2 + 1, X[0] * X[1] - 2))
    domain = msh.ufl_domain()
    closed_form = _eval_on_owned_cells(msh, _INVERSE_PULLBACKS[name](f, domain))
    reference = _eval_on_owned_cells(msh, pullback.apply_inverse(f, domain))
    np.testing.assert_allclose(closed_form, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name, pullback", _PULLBACKS)
def test_apply_pullback_inverse_maps_to_reference_cell(comm, name, pullback):
    # Whichever branch this UFL takes, the values must be the inverse pullback.
    msh = _distorted_quad_mesh(comm)
    X = ufl_pkg.SpatialCoordinate(msh)
    f = ufl_pkg.as_vector((X[0] ** 2 + 1, X[0] * X[1] - 2))
    domain = msh.ufl_domain()
    result = _eval_on_owned_cells(msh, apply_pullback_inverse(pullback, f, domain))
    expected = _eval_on_owned_cells(msh, _INVERSE_PULLBACKS[name](f, domain))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_apply_pullback_inverse_without_closed_form(comm):
    pullback = ufl_pkg.pullback.double_covariant_piola
    msh = _distorted_quad_mesh(comm)
    f = ufl_pkg.as_matrix(((1.0, 0.0), (0.0, 1.0)))
    if hasattr(pullback, "apply_inverse"):
        assert apply_pullback_inverse(pullback, f, msh.ufl_domain()) is not None
    else:
        with pytest.raises(NotImplementedError, match="apply_inverse"):
            apply_pullback_inverse(pullback, f, msh.ufl_domain())
