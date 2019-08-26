
ROOT:=..
include readies/mk/main

MK_CMAKE:=1
MK_CMAKE_INSTALL:=1

define HELP
make setup    # install prerequisited (CAUTION: THIS WILL MODIFY YOUR SYSTEM)
make fetch    # download and prepare dependant modules
make build    # compile and link
make clean    # remove build artifacts
make test     # run tests
make pack     # create installation packages
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
# INSTALL_DIR=$(ROOT)/install-$(DEVICE)
DEPS_DIR=$(ROOT)/deps/$(OS)-$(ARCH)-$(DEVICE)
INSTALL_DIR=$(ROOT)/bin/$(OS)-$(ARCH)-$(DEVICE)/install

TARGET=$(BINDIR)/redisai.so

BACKENDS_PATH ?= $(INSTALL_DIR)/backends

CMAKE_FLAGS += \
	-DDEPS_PATH=$(realpath $(DEPS_DIR)) \
	-DINSTALL_PATH=$(realpath $(INSTALL_DIR)) \
	-DDEVICE=$(DEVICE)

include $(MK)/defs

#----------------------------------------------------------------------------------------------

.PHONY: deps fetch pack pack_ramp pack_deps test

include $(MK)/rules

#---------------------------------------------------------------------------------------------- 

prebuild:
	$(SHOW)if [ ! -d $(ROOT)/deps ]; then echo $$'Dependencies are not in place.\nPlease run \'make fetch\'.'; exit 1; fi

$(TARGET): prebuild $(MK_MAKEFILES) $(DEPS)
	$(SHOW)mkdir -p $(INSTALL_DIR)
	$(SHOW)$(MAKE) -C $(BINDIR)
	$(SHOW)$(MAKE) -C $(BINDIR) install
#	$(SHOW)cd $(ROOT) ;\
#	if [ ! -e install ]; then ln -sf install-$(DEVICE) install; fi

clean:
ifeq ($(ALL),1)
	$(SHOW)cd $(ROOT) ;\
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

TEST_REPORT_DIR ?= $(PWD)

test:
	$(SHOW)if [ "$(git lfs env > /dev/null 2>&1 ; echo $?)" != "0" ]; then git lfs install; fi
	$(SHOW)git lfs pull
	$(SHOW)set -e ;\
	cd $(ROOT)/test ;\
	python3 -m RLTest $(TEST_ARGS) --test basic_tests.py \
		--junitxml=$(TEST_REPORT_DIR)/tests_report.xml \
		--module $(INSTALL_DIR)/redisai.so
