#!/usr/bin/env bash

###### SET VERSIONS ######

ORT_VERSION="1.9.0"
DLPACK_VERSION="v0.5_RAI"
TF_VERSION="2.6.0"
TFLITE_VERSION="2.0.0"
PT_VERSION="1.11.0"

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
		VERBOSE=1          Print commands
		FORCE=1            Download even if present
		WITH_DLPACK=0      Skip dlpack
		WITH_TF=0          Skip Tensorflow or download official version
		WITH_TFLITE=0      Skip TensorflowLite or download official version
		WITH_PT=0          Skip PyTorch or download official version
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


# Get the backend from its URL if is not already found and unpack it
clean_and_fetch() {
    product=$1
    archive=$2
    src_url=$3
    no_fetch=$4

	[[ $FORCE == 1 ]] && rm -rf ${product}  # FORCE is from the env
    [[ $FORCE != 1 ]] && [[ -d ${product} ]]  && echo "${product} is in place, skipping. Set FORCE=1 to override. Continuing." && return
	echo "Installing ${product} from ${src_url} in `pwd`..."
	[[ ! -e ${archive} ]] && [[ -z ${no_fetch} ]] && wget -q ${src_url}
	rm -rf ${product}.x
	mkdir ${product}.x
	tar xzf ${archive} --no-same-owner --strip-components=1 -C ${product}.x
	mv ${product}.x ${product}
    echo "Done."
}

# This is for torch backend, which comes in a zip file
clean_and_fetch_torch() {
  archive=$1
  src_url=$2

  [[ $FORCE == 1 ]] && rm -rf libtorch  # FORCE is from the env
  [[ $FORCE != 1 ]] && [[ -d libtorch ]]  && echo "libtorch is in place, skipping. Set FORCE=1 to override. Continuing." && return
  echo "Installing libtorch from ${src_url} in `pwd`..."
  LIBTORCH_ZIP=libtorch-${DEVICE}-${PT_VERSION}.zip
  wget -q -O ${LIBTORCH_ZIP} ${src_url}
  unzip -q -o ${LIBTORCH_ZIP}
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
    else
        echo "Only x64 is supported currently"
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

LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
if [[ $OS == linux ]]; then
    TFLITE_OS="linux"
    if [[ $ARCH == x64 ]]; then
        TFLITE_ARCH=x86_64
    fi
elif [[ $OS == macos ]]; then
        TFLITE_OS=darwin
        # TFLITE_BUILD=cpu
        TFLITE_ARCH=x86_64
fi

LIBTFLITE_ARCHIVE=libtensorflowlite-${TFLITE_OS}-${TFLITE_ARCH}-${TFLITE_VERSION}.tar.gz

if [[ $WITH_TFLITE != 0 ]]; then
    clean_and_fetch libtensorflow-lite ${LIBTFLITE_ARCHIVE} ${LIBTF_URL_BASE}/${LIBTFLITE_ARCHIVE}
else
	echo "Skipping TensorFlow Lite."
fi # WITH_TFLITE

####################################################################################### LIBTORCH

PT_BUILD=cpu
if [[ $OS == linux ]]; then
    PT_OS=linux
    if [[ $GPU == 1 ]]; then
        PT_BUILD=cu113
    fi
    if [[ $ARCH == x64 ]]; then
        PT_ARCH=x86_64
    else
        echo "Only x64 is supported currently"
    fi
    LIBTORCH_ARCHIVE=libtorch-cxx11-abi-shared-with-deps-${PT_VERSION}%2B${PT_BUILD}.zip

elif [[ $OS == macos ]]; then
    PT_OS=macos
    PT_ARCH=x86_64
    PT_BUILD=cpu
    PT_REPACK=1
    LIBTORCH_ARCHIVE=libtorch-macos-${PT_VERSION}.zip
fi


LIBTORCH_URL=https://download.pytorch.org/libtorch/$PT_BUILD/$LIBTORCH_ARCHIVE

if [[ $WITH_PT != 0 ]]; then
    clean_and_fetch_torch ${LIBTORCH_ARCHIVE} ${LIBTORCH_URL}
else
	echo "Skipping libtorch."
fi # WITH_PT

############################################################################# ONNX

ORT_URL_BASE=https://s3.amazonaws.com/redismodules/onnxruntime
ORT_BUILD=""
if [[ $OS == linux ]]; then
    ORT_OS=linux
    if [[ $GPU == 1 ]]; then
        ORT_BUILD="-gpu"
    fi
    if [[ $ARCH == x64 ]]; then
        ORT_ARCH=x64
    else
        echo "Only x64 is supported currently"
    fi
elif [[ $OS == macos ]]; then
    ORT_OS=osx
    ORT_ARCH=x64
    ORT_BUILD=""
    ORT_URL_BASE=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}
fi

ORT_ARCHIVE=onnxruntime-${ORT_OS}-${ORT_ARCH}${ORT_BUILD}-${ORT_VERSION}.tgz
if [[ $WITH_ORT != 0 ]]; then
    clean_and_fetch onnxruntime ${ORT_ARCHIVE}  ${ORT_URL_BASE}/${ORT_ARCHIVE}
else
	echo "Skipping ONNXRuntime."
fi # WITH_ORT
