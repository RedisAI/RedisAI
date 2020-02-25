
ROOT:=..
include readies/mk/main

MK.cmake:=1
MK_CUSTOM_CLEAN:=1

MK_ALL_TARGETS=bindirs fetch build pack

#---------------------------------------------------------------------------------------------- 

define HELP
make setup  # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch  # download and prepare dependant modules

make build  # compile and link
  DEBUG=1          # build for debugging 
  VARIANT=name     # build variant `name`
  WHY=1            # explain CMake decisions (into /tmp/cmake.why) 
make clean  # remove build artifacts
  ALL=1       # remove entire artifacts directory

make test         # run tests
  TEST=test         # run only test `test` with Redis output
  TEST_ARGS=args    # add extra RLTest `args`
  VERBOSE=1         # verbose tests output
  COV=1             # perform coverage analysis
make cov-upload   # upload coverage data to codecov.io (requires CODECOV_TOKEN)

make pack       # create installation packages
  PACK_DEPS=0     # do not pack dependencies
  INTO=dir        # place artifacts in `dir`
  BRANCH=name     # use `name` as branch name
make deploy     # copy packages to S3
make release    # release a version

fetch and build options:
  WITH_TF=0     # SKip TensofFlow
  WITH_TFLITE=0 # SKip TensofFlowLite
  WITH_PT=0     # Skip PyTorch
  WITH_ORT=0    # SKip ONNXRuntime

device selection options (fetch, build, and test):
  CPU=1         # build for CPU
  GPU=1         # build for GPU
  CUDA=1        # build for GPU
endef

#---------------------------------------------------------------------------------------------- 

override GPU:=$(or $(findstring $(CUDA),1),$(findstring $(GPU),1))

ifeq ($(GPU),1)
ifeq ($(CPU),1)
$(error CPU=1 and GPU=1 (or CUDA=1) are conflicting)
endif
DEPS_FLAGS=gpu
DEVICE=gpu
else
DEPS_FLAGS=cpu
DEVICE=cpu
endif

#---------------------------------------------------------------------------------------------- 

SRCDIR=..
BINDIR=$(BINROOT)/src
DEPS_DIR=$(ROOT)/deps/$(OS)-$(ARCH)-$(DEVICE)
INSTALL_DIR=$(BINROOT)/install-$(DEVICE)

TARGET=$(BINDIR)/redisai.so
INSTALLED_TARGET=$(INSTALL_DIR)/redisai.so

BACKENDS_PATH ?= $(INSTALL_DIR)/backends

REDIS_VALGRIND_SUPRESS=$(ROOT)/opt/redis_valgrind.sup

#---------------------------------------------------------------------------------------------- 

CMAKE_FILES += \
	$(SRCDIR)/CMakeLists.txt \
	$(SRCDIR)/src/CMakeLists.txt \
	$(SRCDIR)/libtorch_c/CMakeLists.txt

ifeq ($(COV),1)
USE_COVERAGE=on
else
USE_COVERAGE=off
endif

CMAKE_FLAGS += \
	-DDEPS_PATH=$(abspath $(DEPS_DIR)) \
	-DINSTALL_PATH=$(abspath $(INSTALL_DIR)) \
	-DUSE_COVERAGE=$(USE_COVERAGE) \
	-DDEVICE=$(DEVICE)

ifeq ($(WITH_TF),0)
CMAKE_FLAGS += -DBUILD_TF=off
endif

ifeq ($(WITH_TFLITE),0)
CMAKE_FLAGS += -DBUILD_TFLITE=off
endif

ifeq ($(WITH_PT),0)
CMAKE_FLAGS += -DBUILD_TORCH=off
endif

ifeq ($(WITH_ORT),0)
CMAKE_FLAGS += -DBUILD_ORT=off
endif

include $(MK)/defs

#----------------------------------------------------------------------------------------------

.PHONY: deps prebuild fetch pack pack_ramp pack_deps test

include $(MK)/rules

#---------------------------------------------------------------------------------------------- 

prebuild:
	$(SHOW)if [ ! -d $(DEPS_DIR) ]; then echo $$'Dependencies are not in place.\nPlease run \'make fetch\'.'; exit 1; fi

$(info $(TARGET): $(MK_MAKEFILES) $(DEPS))
$(TARGET): $(DEPS)
	$(SHOW)mkdir -p $(INSTALL_DIR)
	$(SHOW)$(MAKE) -C $(BINDIR)
	$(SHOW)$(MAKE) -C $(BINDIR) install
#	$(SHOW)cd $(ROOT) ;\
#	if [ ! -e install ]; then ln -sf install-$(DEVICE) install; fi

install:
	$(SHOW)mkdir -p $(INSTALL_DIR)
	$(SHOW)$(MAKE) -C $(BINDIR) install

clean:
ifeq ($(ALL),1)
	$(SHOW)if [ -d "$(BINROOT)" ]; then rm -rf $(BINROOT); fi
	$(SHOW)if [ -d "$(INSTALL_DIR)" ]; then rm -rf $(INSTALL_DIR); fi
	$(SHOW)rm -f $(ROOT)/install-$(DEVICE)
else
	-$(SHOW)$(MAKE) -C $(BINDIR) clean
endif

#---------------------------------------------------------------------------------------------- 

fetch deps:
	@echo Fetching dependencies...
	$(SHOW)VERBOSE=$(_SHOW) $(ROOT)/get_deps.sh $(DEPS_FLAGS)

#----------------------------------------------------------------------------------------------

pack:
ifneq ($(PACK_DEPS),0)
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(BINROOT) INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) INTO=$(INTO) DEPS=1 ./pack.sh
else
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(BINROOT) INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) INTO=$(INTO) DEPS=0 ./pack.sh
endif

#----------------------------------------------------------------------------------------------

export GEN ?= 1
export SLAVES ?= 1
export AOF ?= 1

define VALGRIND_TEST_CMD
$(TEST_CMD) \
	--no-output-catch \
	--use-valgrind \
	--vg-no-fail-on-errors \
	--vg-verbose
	--vg-options "$(VALGRIND_OPTIONS)" \
	--vg-suppressions $(realpath $(REDIS_VALGRIND_SUPRESS))
endef

test:
ifeq ($(COV),1)
	$(COVERAGE_RESET)
endif
	$(SHOW)\
		DEVICE=$(DEVICE) \
		MODULE=$(INSTALLED_TARGET) \
		GEN=$(GEN) AOF=$(AOF) SLAVES=$(SLAVES) \
		$(ROOT)/test/tests.sh
ifeq ($(COV),1)
	$(COVERAGE_COLLECT_REPORT)
endif

#----------------------------------------------------------------------------------------------

docker:
	$(SHOW)docker build -t redisai --build-arg TEST=1 --build-arg PACK=1 ..

#----------------------------------------------------------------------------------------------

VALGRIND_ARGS=\
	$(VALGRIND_OPTIONS) \
	--suppressions=$(realpath $(REDIS_VALGRIND_SUPRESS)) \
	-v redis-server --protected-mode no --save "" --appendonly no

valgrind: $(TARGET)
	$(SHOW)valgrind $(VALGRIND_ARGS) --loadmodule $(realpath $(INSTALLED_TARGET))

CALLGRIND_ARGS=\
	--tool=callgrind \
	--dump-instr=yes \
	--simulate-cache=no \
	--collect-jumps=no \
	--collect-atstart=yes \
	--instr-atstart=yes \
	-v redis-server --protected-mode no --save "" --appendonly no

callgrind: $(TARGET)
	$(SHOW)valgrind $(CALLGRIND_ARGS) --loadmodule $(realpath $(INSTALLED_TARGET))
