# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyLaia
SOURCEDIR     = source
BUILDDIR      = build
CODEDIR       = ../laia

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Usage (for html): make html 
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

clean:
	@echo "Removing everything under 'build'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees

apidoc:
	# Overwrite docs with the code's docstrings
	sphinx-apidoc -fo $(SOURCEDIR) $(CODEDIR)
