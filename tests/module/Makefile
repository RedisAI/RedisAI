
# find the OS
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')

# Compile flags for linux / osx
ifeq ($(uname_S),Linux)
	SHOBJ_CFLAGS ?= -W -Wall -fno-common -g -ggdb -std=c99 -O2
	SHOBJ_LDFLAGS ?= -shared
else
	SHOBJ_CFLAGS ?= -W -Wall -dynamic -fno-common -g -ggdb -std=c99 -O2
	SHOBJ_LDFLAGS ?= -bundle -undefined dynamic_lookup
endif

TEST_MODULES = LLAPI.so

.PHONY: all

all: $(TEST_MODULES)

32bit:
	$(MAKE) CFLAGS="-m32" LDFLAGS="-melf_i386"

%.o: %.c
	$(CC) -I../../src -DREDIS_MODULE_TARGET -DREDISMODULE_EXPERIMENTAL_API $(SHOBJ_CFLAGS) -fPIC -c $< -o $@

%.so: %.o
	$(CC) -o $@ LLAPI.o $(SHOBJ_LDFLAGS) -lc -lm
	chmod +x LLAPI.so

.PHONY: clean

clean:
	rm -f $(TEST_MODULES) $(TEST_MODULES:.so=.xo)
