from packaging.version import Version

from fenicsx_compat._dolfinx_version import DOLFINX_VERSION, at_least, before


def test_dolfinx_version_is_at_least_010():
    assert DOLFINX_VERSION >= Version("0.10.0")


def test_at_least_true_for_lower_bound():
    assert at_least("0.1.0") is True


def test_at_least_false_for_impossible_future_version():
    assert at_least("999.0.0") is False


def test_before_is_the_inverse_of_at_least():
    assert before("999.0.0") is True
    assert before("0.1.0") is False
