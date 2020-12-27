
ROOT:=..
MK.pyver:=3
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

username := $(shell sh -c 'id -u')
usergroup := $(shell sh -c 'id -g')

#----------------------------------------------------------------------------------------------

define HELP
make setup  # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch  # download and prepare dependant artifacts

make build      # compile and link
  DEBUG=1         # build for debugging
  COV=1           # build for coverage analysis (implies DEBUG=1)
  PROFILE=1       # enable profiling compile flags (and debug symbols) for release type.
                  # You can consider this as build type release with debug symbols and -fno-omit-frame-pointer
  VARIANT=name    # build variant `name`
  WHY=1           # explain CMake decisions (into /tmp/cmake.why) 
make clean      # remove build artifacts
  ALL=1           # remove entire artifacts directory
make install    # create ready-to-run scheme (module and engines)

make test        # run test suites (flow an unit)
				 # same as running make unit_tests flow_tests

make unit_tests  # run unit test

make flow_tests  # run flow test
  TEST=test        # run only test `test` with Redis output
  TEST_ARGS=args   # add extra RLTest `args`
  VERBOSE=1        # verbose tests output
  COV=1            # perform coverage analysis
  VALGRIND|VGD=1   # test with Valgrind (implies DEBUG=1)
  CALLGRIND|CGD=1  # test with Callgrind (implies DEBUG=1)
  REDIS=addr       # Address of Redis server for testing

make cov-upload   # upload coverage data to codecov.io (requires CODECOV_TOKEN)

make pack             # create installation packages
  PACK_DEPS=0           # do not pack dependencies
  BRANCH=name           # use `name` as branch name
make deploy           # copy packages to S3
make release          # release a version
make docker           # build docker image
make docker-gpu-test  # run GPU tests

fetch and build options:
  WITH_TF=0           # SKip TensofFlow
  WITH_TFLITE=0       # SKip TensofFlowLite
  WITH_PT=0           # Skip PyTorch
  WITH_ORT=0          # SKip ONNXRuntime
  WITH_UNIT_TESTS=0   # SKip unit tests

device selection options (fetch, build, and test):
  CPU=1         # build for CPU
  GPU=1         # build for GPU
  CUDA=1        # build for GPU
  JETSON=1      # build for Nvidia Jetson
endef

#---------------------------------------------------------------------------------------------- 

override GPU:=$(or $(findstring $(CUDA),1),$(findstring $(GPU),1))

ifeq ($(GPU),1)
ifeq ($(CPU),1)
$(error CPU=1 and GPU=1 (or CUDA=1) are conflicting)
endif
DEPS_FLAGS += GPU=1
DEVICE=gpu
ifeq ($(JETSON),1)
DEPS_FLAGS += JETSON=1
endif
else
DEPS_FLAGS += CPU=1
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

ifeq ($(PROFILE),1)
USE_PROFILE=on
else
USE_PROFILE=off
endif

CMAKE_FLAGS += \
	-DDEPS_PATH=$(abspath $(DEPS_DIR)) \
	-DINSTALL_PATH=$(abspath $(INSTALL_DIR)) \
	-DUSE_COVERAGE=$(USE_COVERAGE) \
	-DUSE_PROFILE=$(USE_PROFILE) \
	-DREDISAI_GIT_SHA=\"$(GIT_SHA)\" \
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

ifeq ($(WITH_UNIT_TESTS),0)
CMAKE_FLAGS += -DPACKAGE_UNIT_TESTS=off
endif

include $(MK)/defs

#----------------------------------------------------------------------------------------------

.PHONY: fetch deps pack pack_ramp pack_deps test format lint

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
	$(SHOW)VERBOSE=$(_SHOW) $(DEPS_FLAGS) $(ROOT)/get_deps.sh

format:
	./clang-format-all.sh

format-docker:
	docker run --rm -v $(abspath $(ROOT))/:/RedisAI/ --user "$(username):$(usergroup)" redislabsmodules/llvm-toolset:latest bash -c "cd RedisAI && make -C opt format"

lint:
	./clang-check-all.sh

lint-docker:
	docker run --rm -v $(abspath $(ROOT))/:/RedisAI/ --user "$(username):$(usergroup)" redislabsmodules/llvm-toolset:latest bash -c "cd RedisAI && make -C opt lint"

#----------------------------------------------------------------------------------------------

pack: $(INSTALLED_TARGET)
	$(SHOW)find $(INSTALL_DIR) -name "*.so" -exec chmod +x {} \;
	$(SHOW)mkdir -p $(ROOT)/bin/artifacts
ifneq ($(PACK_DEPS),0)
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(ROOT)/bin/artifacts INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) RAMP=1 DEPS=1 ./pack.sh
else
	$(SHOW)DEVICE=$(DEVICE) BINDIR=$(ROOT)/bin/artifacts INSTALL_DIR=$(INSTALL_DIR) BRANCH=$(BRANCH) RAMP=1 DEPS=0 ./pack.sh
endif

#----------------------------------------------------------------------------------------------

export GEN ?= 1
export SLAVES ?= 1
export AOF ?= 1
export CLUSTER ?= 1

define UNIT_TESTS
$(SHOW)echo Running unit tests ...
$(SHOW)\
BINDIR=$(realpath $(BINDIR)) \
$(ROOT)/tests/unit/tests.sh
endef

define FLOW_TESTS
$(SHOW)echo Running flow tests ...
$(SHOW)\
DEVICE=$(DEVICE) \
MODULE=$(realpath $(INSTALLED_TARGET)) \
CLUSTER=$(CLUSTER) \
GEN=$(GEN) AOF=$(AOF) SLAVES=$(SLAVES) \
VALGRIND=$(VALGRIND) \
$(ROOT)/tests/flow/tests.sh
endef

unit_tests: build
	$(COVERAGE_RESET)
	$(UNIT_TESTS)
	$(COVERAGE_COLLECT_REPORT)

flow_tests: build
	$(COVERAGE_RESET)
	$(SHOW)\
		DEVICE=$(DEVICE) \
		MODULE=$(realpath $(INSTALLED_TARGET)) \
		CLUSTER=$(CLUSTER) \
		GEN=$(GEN) AOF=$(AOF) SLAVES=$(SLAVES) \
		VALGRIND=$(VALGRIND) \
		REDIS=$(REDIS) \
		$(ROOT)/tests/flow/tests.sh

	$(COVERAGE_COLLECT_REPORT)

test: build
	$(COVERAGE_RESET)
	$(UNIT_TESTS)
	$(SHOW)\
		BINDIR=$(realpath $(BINDIR)) \
		$(ROOT)/tests/unit/tests.sh
	$(SHOW)\
		DEVICE=$(DEVICE) \
		MODULE=$(realpath $(INSTALLED_TARGET)) \
		CLUSTER=$(CLUSTER) \
		GEN=$(GEN) AOF=$(AOF) SLAVES=$(SLAVES) \
		VALGRIND=$(VALGRIND) \
		REDIS=$(REDIS) \
		$(ROOT)/tests/flow/tests.sh
	$(COVERAGE_COLLECT_REPORT)

#----------------------------------------------------------------------------------------------

valgrind:
	$(SHOW)$(ROOT)/tests/flow/valgrind.sh $(realpath $(INSTALLED_TARGET))

callgrind:
	$(SHOW)CALLGRIND=1 $(ROOT)/tests/flow/valgrind.sh $(realpath $(INSTALLED_TARGET))

#----------------------------------------------------------------------------------------------

docker:
	$(SHOW)docker build -t redisai --build-arg TEST=1 --build-arg PACK=1 ..

docker-gpu-test:
	$(SHOW)set -e ;\
		cd .. ;\
		docker build -f Dockerfile.gpu-test -t redisai-test:latest-$(DEVICE)-$(ARCH)-$(OS) .
		docker run --gpus all -it redisai-test:latest-$(DEVICE)-$(ARCH)-$(OS)
