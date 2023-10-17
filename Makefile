.PHONY: release

release:
	$(eval version:=$(shell cat laia/VERSION))
	git commit VERSION -m "Version $(version)"
	git tag $(version)
	git push origin master $(version)
