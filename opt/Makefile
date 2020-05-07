
ROOT:=..
include readies/mk/main

MK.cmake:=1
MK_CUSTOM_CMAKE_BUILD:=1
MK_CUSTOM_CLEAN:=1

MK_ALL_TARGETS=bindirs fetch build pack

ifneq ($(VGD),)
VALGRIND=$(VGD)
endif

ifeq ($(VALGRIND),1)
DEBUG ?= 1
endif

ifeq ($(COV),1)
DEBUG ?= 1
endif

#----------------------------------------------------------------------------------------------

define HELP
make setup  # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch  # download and prepare dependant modules

make build      # compile and link
  DEBUG=1         # build for debugging
  COV=1           # build for coverage analysis (implies DEBUG=1)
  VARIANT=name    # build variant `name`
  VERBOSE=1       # verbose build output
  WHY=1           # explain CMake decisions (into /tmp/cmake.why)
make clean      # remove build artifacts
  ALL=1           # remove entire artifacts directory
make install    # create ready-to-run scheme (module and engines)

make test         # run tests
  TEST=test         # run only test `test` with Redis output
  TEST_ARGS=args    # add extra RLTest `args`
  VERBOSE=1         # verbose tests output
  COV=1             # perform coverage analysis
  VALGRIND|VGD=1    # test with Valgrind (implies DEBUG=1)
  CALLGRIND|CGD=1   # test with Callgrind (implies DEBUG=1)
make cov-upload   # upload coverage data to codecov.io (requires CODECOV_TOKEN)

make pack             # create installation packages
  PACK_DEPS=0           # do not pack dependencies
  INTO=dir              # place artifacts in `dir`
  BRANCH=name           # use `name` as branch name
make deploy           # copy packages to S3
make release          # release a version
make docker           # build docker image
make docker-gpu-test  # run GPU tests

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
COV_EXCLUDE= "*/rmutil/*" "*src/util/*" "*/deps/*"

TARGET=$(BINDIR)/redisai.so
INSTALLED_TARGET=$(INSTALL_DIR)/redisai.so

BACKENDS_PATH ?= $(INSTALL_DIR)/backends

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
	-DDEVICE=$(DEVICE) \
	-DREDISAI_GIT_SHA=$(GIT_SHA)

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

.PHONY: fetch deps pack pack_ramp pack_deps test

include $(MK)/rules

#----------------------------------------------------------------------------------------------

#prebuild:
#	$(SHOW)if [ ! -d $(DEPS_DIR) ]; then echo $$'Dependencies are not in place.\nPlease run \'make fetch\'.'; exit 1; fi

cmake-build $(TARGET): $(MK_MAKEFILES)
	$(SHOW)$(MAKE) -C $(BINDIR)
	$(SHOW)mkdir -p $(INSTALL_DIR)
	$(SHOW)$(MAKE) -C $(BINDIR) install
	$(SHOW)find $(INSTALL_DIR) -name "*.so" -exec chmod +x {} \;

install $(INSTALLED_TARGET): $(TARGET)
	$(SHOW)mkdir -p $(INSTALL_DIR)
	$(SHOW)$(MAKE) -C $(BINDIR) install
	$(SHOW)find $(INSTALL_DIR) -name "*.so" -exec chmod +x {} \;

clean:
ifeq ($(ALL),1)
	$(SHOW)if [ -d "$(BINROOT)" ]; then rm -rf $(BINROOT); fi
	$(SHOW)if [ -d "$(INSTALL_DIR)" ]; then rm -rf $(INSTALL_DIR); fi
	$(SHOW)rm -f $(ROOT)/install-$(DEVICE)
else
	-$(SHOW)$(MAKE) -C $(BINDIR) clean
endif

#----------------------------------------------------------------------------------------------

setup:
	@echo Setting up system...
	$(SHOW)$(ROOT)/opt/readies/bin/getpy3
	$(SHOW)$(ROOT)/opt/system-setup.py

fetch deps:
	@echo Fetching dependencies...
	$(SHOW)VERBOSE=$(_SHOW) $(ROOT)/get_deps.sh $(DEPS_FLAGS)

#----------------------------------------------------------------------------------------------

pack: $(INSTALLED_TARGET)
	$(SHOW)find $(INSTALL_DIR) -name "*.so" -exec chmod +x {} \;
ifneq ($(PACK_DEPS),0)
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(BINROOT) INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) INTO=$(INTO) DEPS=1 ./pack.sh
else
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(BINROOT) INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) INTO=$(INTO) DEPS=0 ./pack.sh
endif

#----------------------------------------------------------------------------------------------

export GEN ?= 1
export SLAVES ?= 1
export AOF ?= 1

test:
	$(COVERAGE_RESET)
	$(SHOW)\
		DEVICE=$(DEVICE) \
		MODULE=$(INSTALLED_TARGET) \
		GEN=$(GEN) AOF=$(AOF) SLAVES=$(SLAVES) \
		VALGRIND=$(VALGRIND) \
		$(ROOT)/test/tests.sh
	$(COVERAGE_COLLECT_REPORT)

valgrind:
	$(SHOW)$(ROOT)/test/valgrind.sh $(realpath $(INSTALLED_TARGET))

callgrind:
	$(SHOW)CALLGRIND=1 $(ROOT)/test/valgrind.sh $(realpath $(INSTALLED_TARGET))

#----------------------------------------------------------------------------------------------

docker:
	$(SHOW)docker build -t redisai --build-arg TEST=1 --build-arg PACK=1 ..

docker-gpu-test:
	$(SHOW)set -e ;\
		cd .. ;\
		docker build -f Dockerfile.gpu-test -t redisai-test:latest-$(DEVICE)-$(ARCH)-$(OS) .
		docker run --gpus all -it redisai-test:latest-$(DEVICE)-$(ARCH)-$(OS)
