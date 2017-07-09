
FINAL_CFLAGS=$(CFLAGS)
FINAL_LDFLAGS=$(LDFLAGS)
FINAL_LDPATH= 
TS_CC=gcc
TS_LD=ld

DEPS_PATH=../deps
INSTALL_PATH=../install/
INSTALL=install

TS_OBJ = redistf.o

INCLUDE_FLAGS=-I $(DEPS_PATH)/redis/src -I $(DEPS_PATH)/tensorflow
LDPATH_FLAGS=-L $(DEPS_PATH)/tensorflow/bazel-bin/tensorflow

ifeq ($(uname_S),Linux)
	FINAL_CFLAGS += -fPIC -std=gnu99
	FINAL_LDFLAGS += -shared -Bsymbolic -lc -ltensorflow_cc
	FINAL_LDPATH += LD_LIBRARY_PATH=$(INSTALL_PATH)
else
	FINAL_CFLAGS += -dynamic -fno-common -std=c99
	FINAL_LDFLAGS += -bundle -undefined dynamic_lookup -lc -ltensorflow_cc
	FINAL_LDPATH += DYLD_LIBRARY_PATH=$(INSTALL_PATH)
endif

all: redistf.so

%.o: %.c
	$(TS_CC) $(INCLUDE_FLAGS) -c $(FINAL_CFLAGS) $<

redistf.so: $(TS_OBJ)
	$(TS_LD) $(LDPATH_FLAGS) $(FINAL_LDFLAGS) -o redistf.so $(TS_OBJ)

clean:
	rm *.o *.so

cleaninst:
	rm -rf $(INSTALL_PATH)

cleandeps: clean
	rm -rf $(DEPS_PATH)

install: all
	@mkdir -p $(INSTALL_PATH)
	$(INSTALL) redistf.so $(INSTALL_PATH)
	$(INSTALL) $(DEPS_PATH)/redis/src/redis-server $(INSTALL_PATH)
	$(INSTALL) $(DEPS_PATH)/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so $(INSTALL_PATH)

run: install
	$(FINAL_LDPATH) $(INSTALL_PATH)/redis-server --loadmodule $(INSTALL_PATH)/redistf.so

