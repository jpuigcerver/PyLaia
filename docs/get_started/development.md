# Development

PyLaia uses different tools during its development.

## Linter

Code syntax is analyzed before submitting the code.

To run the linter tools suite you may use [pre-commit](https://pre-commit.com).

```shell
pip install pre-commit
pre-commit run -a
```

## Tests

### Unit tests

Tests are executed using [tox](https://tox.wiki/en/latest/).

```shell
pip install .[test]
tox
```

## Documentation

This documentation uses [Sphinx](http://www.sphinx-doc.org/) and was generated using [MkDocs](https://mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/).

### Setup

Add the `docs` extra when installing `pylaia`:

```shell
# In a clone of the Git repository
pip install .[docs]
```

Build the documentation using `mkdocs serve -v`. You can then write in [Markdown](https://www.markdownguide.org/) in the relevant `docs/*.md` files, and see live output on http://localhost:8000.
