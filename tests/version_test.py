from pathlib import Path

import laia


def test_version_exists():
    assert hasattr(laia, "__version__")


def test_installed_versions():
    versions = laia.get_installed_versions()
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    with open(requirements_path) as f:
        # +1 for laia's version
        assert len(versions) == len(f.readlines()) + 1
    assert all("==" in v or " @ " in v for v in versions)
