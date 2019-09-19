#!/usr/bin/env bash

set -e
[[ $VERBOSE == 1 ]] && set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

ROOT=$HERE/../../..
. $ROOT/opt/readies/shibumi/functions
ROOT=$(realpath $ROOT)

if [[ "$1" == "cpu" ]]; then
	GPU=no
elif [[ "$1" == "gpu" ]]; then
	GPU=yes
else
	GPU=${GPU:-no}
fi

OS=$(python3 $ROOT/opt/readies/bin/platform --os)
ARCH=$(python3 $ROOT/opt/readies/bin/platform --arch)

# avoid wget warnings on macOS
[[ $OS == macosx ]] && export LC_ALL=en_US.UTF-8

if [[ -z $PT_VERSION ]]; then
	PT_VERSION=1.2.0
	#PT_VERSION="latest"
fi

if [[ $OS == linux ]]; then
	PT_OS=shared-with-deps
	if [[ $GPU == no ]]; then
		PT_BUILD=cpu
	else
		PT_BUILD=cu90
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

LIBTORCH_ARCHIVE=libtorch-${PT_OS}-${PT_VERSION}.zip
[[ -z $LIBTORCH_URL ]] && LIBTORCH_URL=https://download.pytorch.org/libtorch/$PT_BUILD/$LIBTORCH_ARCHIVE

if [ ! -f $LIBTORCH_ARCHIVE ]; then
	echo "Downloading libtorch ${PT_VERSION} ${PT_BUILD}"
	wget -q $LIBTORCH_URL
fi

if [[ $OS == linux ]]; then
	PT_OS=linux
fi

unzip -q -o ${LIBTORCH_ARCHIVE}
tar czf libtorch-${PT_BUILD}-${PT_OS}-${PT_ARCH}-${PT_VERSION}.tar.gz libtorch/
rm -rf libtorch/ ${LIBTORCH_ARCHIVE}
