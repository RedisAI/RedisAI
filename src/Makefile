
FINAL_CFLAGS=$(CFLAGS) -g -DREDISMODULE_EXPERIMENTAL_API
FINAL_LDFLAGS=$(LDFLAGS)
FINAL_LDPATH=
TS_CC=gcc
TS_LD=ld

DEPS_PATH=../deps
INSTALL_PATH=../install/
INSTALL=install

TS_OBJ = redisai.o tensor.o graph.o script.o

INCLUDE_FLAGS=-I .                                  \
              -I $(DEPS_PATH)/redis/src             \
              -I $(DEPS_PATH)/dlpack/include        \

ifeq ($(shell uname),Linux)
	FINAL_CFLAGS += -fPIC -std=gnu99
	FINAL_LDFLAGS += -shared -Bsymbolic -lc
	FINAL_LDPATH += LD_LIBRARY_PATH=$(INSTALL_PATH)
else
	FINAL_CFLAGS += -dynamic -std=c99
	FINAL_LDFLAGS += -bundle -undefined dynamic_lookup -lc
	FINAL_LDPATH += DYLD_LIBRARY_PATH=$(INSTALL_PATH)
endif

# Tensorflow backend
TS_OBJ += backends/tensorflow.o
FINAL_LDFLAGS += -ltensorflow
INCLUDE_FLAGS += -I $(DEPS_PATH)/libtensorflow/include
LDPATH_FLAGS += -L $(DEPS_PATH)/libtensorflow/lib
BACKEND_INSTALL += $(INSTALL) $(DEPS_PATH)/libtensorflow/lib/libtensorflow*.so $(INSTALL_PATH)

# PyTorch backend
TS_OBJ += backends/torch.o
FINAL_LDFLAGS += -ltorch_c
INCLUDE_FLAGS += -I $(DEPS_PATH)/libtorch_c/include
LDPATH_FLAGS += -L $(DEPS_PATH)/libtorch_c/lib
BACKEND_INSTALL += $(INSTALL) $(DEPS_PATH)/libtorch_c/lib/libtorch_c*.so $(INSTALL_PATH)

all: redisai.so

%.o: %.c
	$(TS_CC) $(INCLUDE_FLAGS) -c $(FINAL_CFLAGS) $< -o $@

%.o: backends/%.c
	$(TS_CC) $(INCLUDE_FLAGS) -c $(FINAL_CFLAGS) $< -o backends/$@

redisai.so: $(TS_OBJ)
	$(TS_LD) $(LDPATH_FLAGS) $(FINAL_LDFLAGS) -o redisai.so $(TS_OBJ)

clean:
	rm *.o backends/*.o *.so

cleaninst:
	rm -rf $(INSTALL_PATH)

cleandeps: clean
	rm -rf $(DEPS_PATH)

install: all
	@mkdir -p $(INSTALL_PATH)
	$(INSTALL) redisai.so $(INSTALL_PATH)
	$(INSTALL) $(DEPS_PATH)/redis/src/redis-server $(INSTALL_PATH)
	$(BACKEND_INSTALL)

run: install
	$(FINAL_LDPATH) $(INSTALL_PATH)/redis-server --loadmodule $(INSTALL_PATH)/redisai.so
