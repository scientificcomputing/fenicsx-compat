import pytest

from fenicsx_compat.optional import require_module


def test_require_module_returns_the_module_when_present():
    mod = require_module("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_module_raises_actionable_error_when_missing():
    with pytest.raises(ImportError, match="pip install"):
        require_module("this_module_does_not_exist_anywhere")


def test_require_module_uses_extra_name_in_message():
    with pytest.raises(ImportError, match="pip install some-other-package-name"):
        require_module("this_module_does_not_exist_anywhere", extra="some-other-package-name")
