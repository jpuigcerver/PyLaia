.PHONY: release

release:
	$(eval version:=$(shell cat laia/VERSION))
	git commit laia/VERSION -m "Version $(version)"
	git tag $(version)
	git push origin master $(version)
