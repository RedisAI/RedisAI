
default: all

.DEFAULT:
	cd src && $(MAKE) $@

test:
	cd test && $(MAKE) $@
.PHONY: test

