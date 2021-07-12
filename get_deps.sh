#!/usr/bin/env bash

###### SET VERSIONS ######

ORT_VERSION="1.7.1"
DLPACK_VERSION="v0.5_RAI"
TF_VERSION="2.5.0"
TFLITE_VERSION="2.0.0"
PT_VERSION="1.9.0"

if [[ $JETSON == 1 ]]; then
    PT_VERSION="1.7.0"
    TF_VERSION="2.4.0"
fi

###### END VERSIONS ######

error() {
	echo "There are errors."
	exit 1
}

trap error ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ $1 == --help || $1 == help ]]; then
	cat <<-END
		[ARGVARS...] get_deps.sh [cpu|gpu] [--help|help]

		Argument variables:
		CPU=1              Get CPU dependencies
		GPU=1              Get GPU dependencies
		JETSON=1           Get Jetson dependencies
		VERBOSE=1          Print commands
		FORCE=1            Download even if present
		WITH_DLPACK=0      Skip dlpack
		WITH_TF=0          Skip Tensorflow or download from S3 repo
		WITH_TFLITE=0      Skip TensorflowLite  or download from S3 repo
		WITH_PT=0          Skip PyTorch or download from S3 repo
		WITH_ORT=0         Skip OnnxRuntime or download from S3 repo
		OS=                Set, to override the platform OS
		ARCH=0             Set, to override the platform ARCH

	END
	exit 0
fi

set -e
[[ $VERBOSE == 1 ]] && set -x

# default to cpu
if [[ "$1" == "cpu" || $CPU == 1 ]]; then
	GPU=0
	DEVICE=cpu
elif [[ "$1" == "gpu" || $GPU == 1 ]]; then
	GPU=1
	DEVICE=gpu
else
	GPU=0
	DEVICE=cpu
fi

# default platforms
if [ -f ${HERE}/opt/readies/bin/platform ]; then
    if [ -z ${OS} ]; then
        OS=$(python3 $HERE/opt/readies/bin/platform --os)
    fi
    if [ -z ${ARCH} ]; then
        ARCH=$(python3 $HERE/opt/readies/bin/platform --arch)
    fi
else
    if [ -z ${OS} ]; then
        OS=`uname -s | tr '[:upper:]' '[:lower:]'`
    fi
    if [ -z ${ARCH} ]; then
        uname -m|grep aarch64 || ARCH=x64
        uname -m|grep x86 || ARCH=arm64v8
    fi
fi

# avoid wget warnings on macOS
[[ $OS == macos ]] && export LC_ALL=en_US.UTF-8

DEPS_DIR=$HERE/deps/$OS-$ARCH-$DEVICE
mkdir -p ${DEPS_DIR}
cd ${DEPS_DIR}

clean_and_fetch() {
    product=$1
    archive=$2
    srcurl=$3

	[[ $FORCE == 1 ]] && rm -rf ${product}  # FORCE is from the env
    [[ $FORCE != 1 ]] && [[ -d ${product} ]]  && echo "${product} is in place, skipping. Set FORCE=1 to override. Continuing." && return
	echo "Installing ${product} from ${srcurl} in `pwd`..."
	[[ ! -e ${archive} ]] && wget -q ${srcurl}
	rm -rf ${product}.x
	mkdir ${product}.x
	tar xzf ${archive} --no-same-owner --strip-components=1 -C ${product}.x
	mv ${product}.x ${product}
    echo "Done."
}

######################################################################################### DLPACK
if [[ $WITH_DLPACK != 0 ]]; then
	if [[ ! -d dlpack ]]; then
	    [[ $FORCE == 1 ]] && rm -rf dlpack
	    echo "Fetch dlpack to `pwd` ..."
		git clone -q --depth 1 --branch $DLPACK_VERSION https://github.com/RedisAI/dlpack.git dlpack
		echo "Done."
	else
		echo "dlpack is in place."
	fi
else
	echo "Skipping dlpack..."
fi

################################################################################## LIBTENSORFLOW
#

if [[ $OS == linux ]]; then
    TF_OS="linux"
    if [[ $GPU != 1 ]]; then
        TF_BUILD="cpu"
    else
        TF_BUILD="gpu"
    fi
    if [[ $ARCH == x64 ]]; then
        TF_ARCH=x86_64

        LIBTF_URL_BASE=https://storage.googleapis.com/tensorflow/libtensorflow
    elif [[ $ARCH == arm64v8 ]]; then
        TF_ARCH=arm64
        if [[ $JETSON == 1 ]]; then
            TF_BUILD="gpu-jetson"
        fi
        LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
    elif [[ $ARCH == arm32v7 ]]; then
        TF_ARCH=arm
        LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
    fi
elif [[ $OS == macos ]]; then
    TF_OS=darwin
    TF_BUILD=cpu
    TF_ARCH=x86_64
    if [[ $WITH_TF == S3 ]]; then
        LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
    else
        LIBTF_URL_BASE=https://storage.googleapis.com/tensorflow/libtensorflow
    fi
fi

LIBTF_ARCHIVE=libtensorflow-${TF_BUILD}-${TF_OS}-${TF_ARCH}-${TF_VERSION}.tar.gz

if [[ $WITH_TF != 0 ]]; then
    clean_and_fetch libtensorflow ${LIBTF_ARCHIVE} ${LIBTF_URL_BASE}/${LIBTF_ARCHIVE}
else
	echo "Skipping TensorFlow."
fi # WITH_TF

################################################################################## LIBTFLITE
#

LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
if [[ $OS == linux ]]; then
    TFLITE_OS="linux"
    if [[ $ARCH == x64 ]]; then
        TFLITE_ARCH=x86_64
    elif [[ $ARCH == arm64v8 ]]; then
        TFLITE_ARCH=arm64
    elif [[ $ARCH == arm32v7 ]]; then
        TFLITE_ARCH=arm
    fi
elif [[ $OS == macos ]]; then
    TFLITE_OS=darwin
    TFLITE_ARCH=x86_64
fi

LIBTFLITE_ARCHIVE=libtensorflowlite-${TFLITE_OS}-${TFLITE_ARCH}-${TFLITE_VERSION}.tar.gz
if [[ $WITH_TFLITE != 0 ]]; then
    clean_and_fetch tflite ${LIBTFLITE_ARCHIVE} ${LIBTF_URL_BASE}/${LIBTFLITE_ARCHIVE}
else
	echo "Skipping TensorFlow Lite."
fi # WITH_TFLITE

####################################################################################### LIBTORCH
PT_REPACK=0
PT_BUILD=cpu
PT_ARCH=x86_64
if [[ $OS == linux ]]; then
    PT_OS=linux
    if [[ $GPU == 1 ]]; then
        PT_BUILD=cu110
    fi

    if [[ $ARCH == x64 ]]; then
        PT_REPACK=1
    elif [[ $ARCH == arm64v8 ]]; then
        PT_ARCH=arm64
    elif [[ $ARCH == arm32v7 ]]; then
        PT_ARCH=arm
    fi

    if [[ $JETSON == 1 ]]; then
        PT_BUILD=cu102-jetson
        PT_ARCH=arm64
    fi

elif [[ $OS == macos ]]; then
    PT_OS=macos
    PT_REPACK=1
fi

LIBTORCH_ARCHIVE=libtorch-${PT_BUILD}-${PT_OS}-${PT_ARCH}-${PT_VERSION}.tar.gz

if [[ $PT_REPACK == 1 ]]; then
    echo "Using repack.sh from ${HERE}/opt/build/libtorch/repack.sh"
    PT_VERSION=$PT_VERSION GPU=$GPU OS=${OS} ARCH=${ARCH} $HERE/opt/build/libtorch/repack.sh
else
    LIBTORCH_URL=https://s3.amazonaws.com/redismodules/pytorch/$LIBTORCH_ARCHIVE
fi

if [[ $WITH_PT != 0 ]] && [ $PT_REPACK != 1 ]; then
    clean_and_fetch libtorch ${LIBTORCH_ARCHIVE}  ${LIBTORCH_URL}
else
	echo "SKipping libtorch."
fi # WITH_PT

#############################################################################

ORT_URL_BASE=https://s3.amazonaws.com/redismodules/onnxruntime
ORT_BUILD=""
if [[ $OS == linux ]]; then
    ORT_OS=linux
    if [[ $GPU == 1 ]]; then
        ORT_BUILD="-gpu"
    fi
    if [[ $ARCH == x64 ]]; then
        ORT_ARCH=x64
    elif [[ $ARCH == arm64v8 ]]; then
        ORT_ARCH=arm64
    elif [[ $ARCH == arm32v7 ]]; then
        ORT_ARCH=arm
    fi
elif [[ $OS == macos ]]; then
    ORT_OS=osx
    ORT_ARCH=x64
    ORT_URL_BASE=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}
fi

ORT_ARCHIVE=onnxruntime-${ORT_OS}-${ORT_ARCH}${ORT_BUILD}-${ORT_VERSION}.tgz
if [[ $WITH_ORT != 0 ]]; then
    clean_and_fetch onnxruntime  ${ORT_ARCHIVE}  ${ORT_URL_BASE}/${ORT_ARCHIVE}
else
	echo "Skipping ONNXRuntime."
fi # WITH_ORT
