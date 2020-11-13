from pathlib import Path

import laia


def test_version_exists():
    assert hasattr(laia, "__version__")


def test_installed_versions():
    versions = laia.get_installed_versions()
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    with open(requirements_path) as f:
        expected = len([l for l in f.readlines() if not l.startswith("#")])
        expected += 1  # laia's version
        expected -= 1  # dataclasses
        assert len(versions) == expected
    assert all("==" in v or " @ " in v for v in versions)
