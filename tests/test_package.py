import fenicsx_compat


def test_has_version():
    assert isinstance(fenicsx_compat.__version__, str)
    assert fenicsx_compat.__version__ != ""
