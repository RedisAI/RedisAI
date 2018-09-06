export UNAME_S := $(shell uname -s)

default: all

.DEFAULT:
	cd src && $(MAKE) $@

test:
	cd test && $(MAKE) $@
.PHONY: test

