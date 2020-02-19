
ROOT:=..
include readies/mk/main

MK_CMAKE:=1
MK_CMAKE_INSTALL:=1

define HELP
make setup    # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch    # download and prepare dependant modules
make build    # compile and link
make clean    # remove build artifacts
  ALL=1         # remove entire artifacts directory
make test     # run tests
  TEST=test       # run only test `test` with Redis output
  TEST_ARGS=args  # add extra RLTest `args`
  VERBOSE=1   # verbose tests output
make pack     # create installation packages
  PACK_DEPS=0   # do not pack dependencies
  INTO=dir      # place artifacts in `dir`
  BRANCH=name   # use `name` as branch name
make deploy   # copy packages to S3
make release  # release a version

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

SRCDIR=..
BINDIR=$(BINROOT)/src
# INSTALL_DIR=$(ROOT)/install-$(DEVICE)
DEPS_DIR=$(ROOT)/deps/$(OS)-$(ARCH)-$(DEVICE)
INSTALL_DIR=$(ROOT)/bin/$(OS)-$(ARCH)-$(DEVICE)/install

TARGET=$(BINDIR)/redisai.so

BACKENDS_PATH ?= $(INSTALL_DIR)/backends

CMAKE_FILES += \
	$(SRCDIR)/CMakeLists.txt \
	$(SRCDIR)/src/CMakeLists.txt \
	$(SRCDIR)/libtorch_c/CMakeLists.txt

CMAKE_FLAGS += \
	-DDEPS_PATH=$(abspath $(DEPS_DIR)) \
	-DINSTALL_PATH=$(abspath $(INSTALL_DIR)) \
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

.PHONY: deps fetch pack pack_ramp pack_deps test

include $(MK)/rules

#---------------------------------------------------------------------------------------------- 

prebuild:
	$(SHOW)if [ ! -d $(DEPS_DIR) ]; then echo $$'Dependencies are not in place.\nPlease run \'make fetch\'.'; exit 1; fi

$(TARGET): prebuild $(MK_MAKEFILES) $(DEPS)
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

TEST_REPORT_DIR ?= $(PWD)
ifeq ($(VERBOSE),1)
TEST_ARGS += -v
endif
ifeq ($(TEST),)
TEST=basic_tests.py
PYDEBUG=
else
TEST_ARGS += -s
PYDEBUG=1
endif

TEST_PREFIX=set -e; cd $(ROOT)/test
TEST_CMD=\
	DEVICE=$(DEVICE) PYDEBUG=$(PYDEBUG) \
	python3 -m RLTest $(TEST_ARGS) --test $(TEST) --module $(INSTALL_DIR)/redisai.so

GEN ?= 1
SLAVES ?= 1
AOF ?= 1

test:
ifneq ($(NO_LFS),1)
	$(SHOW)if [ "$(git lfs env > /dev/null 2>&1 ; echo $?)" != "0" ]; then cd $(ROOT); git lfs install; fi
	$(SHOW)cd $(ROOT); git lfs pull
endif
ifeq ($(GEN),1)
	$(SHOW)$(TEST_PREFIX); $(TEST_CMD)
endif
ifeq ($(AOF),1)
	$(SHOW)$(TEST_PREFIX); printf "\nTests with --use-aof:\n\n" ;\
	$(TEST_CMD) --use-aof
endif
ifeq ($(SLAVES),1)
	$(SHOW)$(TEST_PREFIX); printf "\nTests with --use-slaves:\n\n" ;\
	$(TEST_CMD) --use-slaves
endif

#----------------------------------------------------------------------------------------------

docker:
	$(SHOW)docker build -t redisai --build-arg TEST=1 --build-arg PACK=1 ..
