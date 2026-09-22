import dolfinx.fem
import dolfinx.mesh
import ufl

from fenicsx_compat.petsc import pack_coefficients, pack_constants


def test_pack_constants_and_coefficients_on_a_simple_form(comm):
    msh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    c = dolfinx.fem.Constant(msh, 2.0)
    L = dolfinx.fem.form(c * v * ufl.dx)
    constants = pack_constants(L)
    coeffs = pack_coefficients(L)
    assert constants.shape[0] == 1
    assert constants[0] == 2.0
    assert isinstance(coeffs, dict)
