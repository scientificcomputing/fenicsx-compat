import ufl


def domain_of(expr):
    """Get the single UFL domain an expression is defined on.

    `ufl.domain.extract_domains` is available across the whole 0.10+
    floor; the `expr.ufl_domain()` fallback found duplicated across the
    source repos (some with a fallback, some without — see spec §5.6) is
    below-floor legacy and is not ported.
    """
    domains = ufl.domain.extract_domains(expr)
    if len(domains) != 1:
        raise ValueError(f"Expected exactly one domain, got {len(domains)}")
    return domains[0]


def _inverse_identity(expr, domain):
    return expr


def _inverse_contravariant_piola(expr, domain):
    """`u = J u_ref / det J`, inverted: `u_ref = det J K u`."""
    return ufl.JacobianDeterminant(domain) * ufl.dot(ufl.JacobianInverse(domain), expr)


def _inverse_covariant_piola(expr, domain):
    """`u = K^T u_ref`, inverted: `u_ref = J^T u`."""
    return ufl.dot(ufl.transpose(ufl.Jacobian(domain)), expr)


_INVERSE_PULLBACKS = {
    "IdentityPullback": _inverse_identity,
    "ContravariantPiola": _inverse_contravariant_piola,
    "CovariantPiola": _inverse_covariant_piola,
}


def apply_pullback_inverse(pullback, expr, domain):
    """Map `expr` from the physical cell to the reference cell.

    Uses `pullback.apply_inverse`, added in FEniCS/ufl PR #511 (first
    released in UFL 2026.2.0), when the installed UFL has it. The UFL
    releases dolfinx 0.10 and 0.11 pin (2025.2, 2026.1) lack it, so a
    closed form is used for the identity,
    contravariant Piola and covariant Piola pullbacks; any other pullback
    raises `NotImplementedError` there. Dispatches on `hasattr`.
    """
    if hasattr(pullback, "apply_inverse"):
        return pullback.apply_inverse(expr, domain)
    name = type(pullback).__name__
    try:
        return _INVERSE_PULLBACKS[name](expr, domain)
    except KeyError:
        raise NotImplementedError(
            f"This UFL has no {name}.apply_inverse (added in FEniCS/ufl PR #511), and "
            "fenicsx-compat has no closed form for it. Upgrade UFL."
        ) from None
