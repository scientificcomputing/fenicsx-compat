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
