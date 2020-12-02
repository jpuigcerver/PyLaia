import laia.common.logging as log


def test_filepath(tmpdir):
    filepath = tmpdir / "test"
    log.config(filepath=filepath)
    log.info("test!")
    log.clear()
    assert filepath.exists()


def test_filename(tmpdir):
    with tmpdir.as_cwd():
        filepath = "test"
        log.config(filepath=filepath)
        log.info("test!")
        log.clear()
    assert tmpdir.join(filepath).exists()
