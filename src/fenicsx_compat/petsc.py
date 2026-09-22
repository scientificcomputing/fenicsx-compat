import dolfinx.cpp.fem
import dolfinx.fem
import numpy.typing as npt


def pack_constants(form: dolfinx.fem.Form) -> npt.NDArray:
    """Pack a form's constants, across the public-API-vs-cpp-object split."""
    try:
        return dolfinx.fem.pack_constants(form)
    except AttributeError:
        return dolfinx.cpp.fem.pack_constants(form._cpp_object)


def pack_coefficients(form: dolfinx.fem.Form) -> dict:
    """Pack a form's coefficients, across the public-API-vs-cpp-object split."""
    try:
        return dolfinx.fem.pack_coefficients(form)
    except AttributeError:
        return dolfinx.cpp.fem.pack_coefficients(form._cpp_object)
