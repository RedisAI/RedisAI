#!/usr/bin/env bash

set -e
[[ $VERBOSE == 1 ]] && set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

ROOT=$HERE/../../..
. $ROOT/opt/readies/shibumi/functions
ROOT=$(realpath $ROOT)

if [[ "$1" == "cpu" ]]; then
	GPU=0
elif [[ "$1" == "gpu" ]]; then
	GPU=1
else
	GPU=${GPU:-0}
fi

OS=$(python3 $ROOT/opt/readies/bin/platform --os)
ARCH=$(python3 $ROOT/opt/readies/bin/platform --arch)

# avoid wget warnings on macOS
[[ $OS == macosx ]] && export LC_ALL=en_US.UTF-8

if [[ -z $PT_VERSION ]]; then
	PT_VERSION="latest"
fi

if [[ $OS == linux ]]; then
	PT_OS=shared-with-deps
	if [[ $GPU != 1 ]]; then
		PT_BUILD=cpu
	else
		PT_BUILD=cu101
	fi
	if [[ $ARCH == x64 ]]; then
		PT_ARCH=x86_64
	fi
elif [[ $OS == macosx ]]; then
	PT_OS=macos
	PT_ARCH=x86_64
	PT_BUILD=cpu
fi

[[ "$PT_VERSION" == "latest" ]] && PT_BUILD=nightly/${PT_BUILD}

if [[ $OS == linux ]]; then
	if [[ $PT_VERSION == 1.2.0 ]]; then
		LIBTORCH_ARCHIVE=libtorch-${PT_OS}-${PT_VERSION}.zip
	elif [[ $PT_VERSION == latest ]]; then
		LIBTORCH_ARCHIVE=libtorch-shared-with-deps-latest.zip
	else
		if [[ $GPU != 1 ]]; then
			LIBTORCH_ARCHIVE=libtorch-cxx11-abi-shared-with-deps-${PT_VERSION}%2Bcpu.zip
		else
			LIBTORCH_ARCHIVE=libtorch-cxx11-abi-shared-with-deps-${PT_VERSION}.zip
		fi
	fi
elif [[ $OS == macosx ]]; then
	LIBTORCH_ARCHIVE=libtorch-${PT_OS}-${PT_VERSION}.zip
fi

[[ -z $LIBTORCH_URL ]] && LIBTORCH_URL=https://download.pytorch.org/libtorch/$PT_BUILD/$LIBTORCH_ARCHIVE

LIBTORCH_ZIP=libtorch-${PT_BUILD}-${PT_OS}-${PT_ARCH}-${PT_VERSION}.zip
if [ ! -f $LIBTORCH_ARCHIVE ]; then
	echo "Downloading libtorch ${PT_VERSION} ${PT_BUILD}"
	wget -q -O $LIBTORCH_ZIP $LIBTORCH_URL
fi

if [[ $OS == linux ]]; then
	PT_OS=linux
fi

unzip -q -o $LIBTORCH_ZIP
tar czf libtorch-${PT_BUILD}-${PT_OS}-${PT_ARCH}-${PT_VERSION}.tar.gz libtorch/
rm -rf libtorch/ $LIBTORCH_ZIP
