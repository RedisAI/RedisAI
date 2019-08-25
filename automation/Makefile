
ROOT:=..
include readies/mk/main

MK_CMAKE:=1
MK_CMAKE_INSTALL:=1

define HELP
make setup    # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch    # download and prepare dependant modules
make build    # compile and link
make clean    # remove build artifacts
make pack     # create installation packages
make test     # run tests
make deploy   # copy packages to S3
make release  # release a version
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
INSTALL_DIR=$(ROOT)/install-$(DEVICE)

TARGET=$(BINDIR)/redisai.so

BACKENDS_PATH ?= $(INSTALL_DIR)/backends

CMAKE_FLAGS += \
	-DDEPS_PATH=$(realpath $(ROOT)/deps) \
	-DDEVICE=$(DEVICE)

include $(MK)/defs

#----------------------------------------------------------------------------------------------

.PHONY: deps fetch pack pack_ramp pack_deps test

include $(MK)/rules

#---------------------------------------------------------------------------------------------- 

prebuild:
	@if [ -d $(ROOT)/deps ]; then echo $$'Dependencies are not in place.\nPlease run \'make fetch\'.'; exit 1; fi

$(TARGET): prebuild $(MK_MAKEFILES) $(DEPS)
	$(MAKE) -C $(BINDIR)
	$(MAKE) -C $(BINDIR) install
	cd $(ROOT) ;\
	if [ ! -e install ]; then ln -sf install-$(DEVICE) install; fi

clean:
ifeq ($(ALL),1)
	cd $(ROOT) ;\
	rm -rf build install deps/dlpack deps/install-$(DEVICE) deps/*.tar.gz deps/*.zip deps/*.tgz
else
	$(SHOW)$(MAKE) -C $(BINDIR) clean
endif

#---------------------------------------------------------------------------------------------- 

fetch:
	@echo Fetching dependencies...
	$(SHOW)VERBOSE=$(_SHOW) $(ROOT)/get_deps.sh $(DEPS_FLAGS)

#----------------------------------------------------------------------------------------------

pack: $(BINDIR)/BINDIR pack_ramp pack_deps

$(BINDIR)/BINDIR: $(BIN_DIRS)
	$(SHOW)echo $(BINDIR)>$(BINDIR)/BINDIR

pack_ramp:
	$(SHOW)RAMP=1 DEVICE=$(DEVICE) BINDIR=$(BINROOT) ./pack.sh

pack_deps: pack_ramp
	$(SHOW)DEPS=1 DEVICE=$(DEVICE) BINDIR=$(BINROOT) ./pack.sh

#----------------------------------------------------------------------------------------------

test:
	@[[ $(git lfs env > /dev/null 2>&1 ; echo $?) != 0 ]] && git lfs install
	@git lfs pull
	@set -e ;\
	cd $(ROOT)/test ;\
	python3 -m RLTest $(TEST_ARGS) --test basic_tests.py --module $(INSTALL_DIR)/redisai.so
