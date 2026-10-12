import re
from pathlib import Path

import fenicsx_compat

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide"


def _documented_names() -> set[str]:
    """Names with an entry in the function guide, i.e. a heading of the form ### `name`."""
    names: set[str] = set()
    for page in GUIDE.glob("*.md"):
        names.update(re.findall(r"^#{2,4} `(\w+)`", page.read_text(), flags=re.MULTILINE))
    return names


def test_every_public_function_has_a_guide_entry():
    public = set(fenicsx_compat.__all__) - {"__version__"}
    missing = sorted(public - _documented_names())
    assert not missing, f"No entry in docs/guide for: {missing}"


def test_the_guide_documents_only_public_functions():
    unknown = sorted(_documented_names() - set(fenicsx_compat.__all__))
    assert not unknown, f"docs/guide has entries for names not in fenicsx_compat.__all__: {unknown}"
