# Contributing

Contributions are welcome! Either by reporting bugs, requesting features, or even creating a pull request yourself.

Use this recipe to get ready to work on PyLaia:

```console
# clone PyLaia
git clone https://github.com/jpuigcerver/PyLaia
cd PyLaia

# use a clean Python environment.
# you can skip this if you prefer conda
virtualenv laia-env
source laia-env/bin/activate

# install all dependencies in editable mode,
# including those for development and testing
pip install --editable ".[dev,test]"

# set-up pre-commit hooks
pre-commit install
```

You can run the test suite (including a coverage report) with:

```console
pytest --cov=laia tests
```

Do not worry about code formatting, `pre-commit` will do the work for you when you try to commit. You can also run it manually with:

```console
pre-commit run --all-files
```

Commits and pull requests are tested using GitHub actions CI, so you don't have to worry about testing each Python and PyTorch version combination.
