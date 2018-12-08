
FINAL_CFLAGS=$(CFLAGS) -g -DREDISMODULE_EXPERIMENTAL_API
FINAL_LDFLAGS=$(LDFLAGS)
FINAL_LDPATH=
TS_CC=gcc
TS_LD=ld

DEPS_PATH=../deps
INSTALL_PATH=../install/
INSTALL=install

TS_OBJ = redisdl.o tensor.o graph.o

INCLUDE_FLAGS=-I $(DEPS_PATH)/redis/src             \
              -I $(DEPS_PATH)/dlpack/include        \
			  -I $(DEPS_PATH)/libtensorflow/include
LDPATH_FLAGS=-L $(DEPS_PATH)/libtensorflow/lib

ifeq ($(shell uname),Linux)
	FINAL_CFLAGS += -fPIC -std=gnu99
	FINAL_LDFLAGS += -shared -Bsymbolic -lc -ltensorflow
	FINAL_LDPATH += LD_LIBRARY_PATH=$(INSTALL_PATH)
else
	FINAL_CFLAGS += -dynamic -std=c99
	FINAL_LDFLAGS += -bundle -undefined dynamic_lookup -lc -ltensorflow
	FINAL_LDPATH += DYLD_LIBRARY_PATH=$(INSTALL_PATH)
endif

all: redisdl.so

%.o: %.c
	$(TS_CC) $(INCLUDE_FLAGS) -c $(FINAL_CFLAGS) $<

redisdl.so: $(TS_OBJ)
	$(TS_LD) $(LDPATH_FLAGS) $(FINAL_LDFLAGS) -o redisdl.so $(TS_OBJ)

clean:
	rm *.o *.so

cleaninst:
	rm -rf $(INSTALL_PATH)

cleandeps: clean
	rm -rf $(DEPS_PATH)

install: all
	@mkdir -p $(INSTALL_PATH)
	$(INSTALL) redisdl.so $(INSTALL_PATH)
	$(INSTALL) $(DEPS_PATH)/redis/src/redis-server $(INSTALL_PATH)
	$(INSTALL) $(DEPS_PATH)/libtensorflow/lib/libtensorflow*.so $(INSTALL_PATH)

run: install
	$(FINAL_LDPATH) $(INSTALL_PATH)/redis-server --loadmodule $(INSTALL_PATH)/redisdl.so

