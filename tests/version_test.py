import pkg_resources

from laia import *  # pylint: disable=wildcard-import


def test_wildcard_import():
    # these are defined in __all__
    assert __version__
    assert __root__
    assert get_installed_versions


def test_versions_match():
    # check __init__ version matches setup.py (installed) version
    version = pkg_resources.require("pylaia")[0].version
    print(version)
    print(__version__)
    assert __version__.startswith(version)


def test_installed_versions():
    versions = get_installed_versions()
    requirements_path = __root__ / "requirements.txt"
    if not requirements_path.exists():
        assert not versions
    else:
        with open(requirements_path) as f:
            expected = len([l for l in f.readlines() if not l.startswith("#")])
            expected += 1  # laia's version
            expected -= 1  # dataclasses
            assert len(versions) == expected
        assert all("==" in v or " @ " in v for v in versions)
