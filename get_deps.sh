#!/usr/bin/env bash

set -e
set -x

BASE_DIRECTORY=`pwd`

# Allow different deps for different platforms:

if [ -z "$DEPS_DIRECTORY" ]; then
  DEPS_DIRECTORY=${BASE_DIRECTORY}/deps
fi

mkdir -p ${DEPS_DIRECTORY}
cd ${DEPS_DIRECTORY}
DEPS_DIRECTORY=$PWD # -- to avoid relative/absolute confusion

PREFIX=${DEPS_DIRECTORY}/install
mkdir -p ${PREFIX}
rm -rf ${PREFIX}/share
rm -rf ${PREFIX}/lib
rm -rf ${PREFIX}/include

if [ ! -d dlpack ]; then
    echo "Cloning dlpack"
    git clone --depth 1 https://github.com/dmlc/dlpack.git
fi

mkdir -p ${PREFIX}
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  cp -d -r --no-preserve=ownership dlpack/include ${PREFIX}
elif [[ "$OSTYPE" == "darwin"* ]]; then
  cp -r dlpack/include ${PREFIX}
fi

## TENSORFLOW
TF_VERSION="1.12.0"
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  TF_OS="linux"
  if [[ "$1" == "cpu" ]]; then
    TF_BUILD="cpu"
  else
    TF_BUILD="gpu"
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  TF_OS="darwin"
  TF_BUILD="cpu"
fi

LIBTF_ARCHIVE=libtensorflow-${TF_BUILD}-${TF_OS}-x86_64-${TF_VERSION}.tar.gz

if [ ! -e ${LIBTF_ARCHIVE} ]; then
  echo "Downloading libtensorflow ${TF_VERSION} ${TF_BUILD}"
  wget https://storage.googleapis.com/tensorflow/libtensorflow/${LIBTF_ARCHIVE}
fi

tar xf ${LIBTF_ARCHIVE} --no-same-owner --strip-components=1 -C ${PREFIX}

## PYTORCH

PT_VERSION="1.0.1"
#PT_VERSION="latest"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  PT_OS="shared-with-deps"
  if [[ "$1" == "cpu" ]]; then
    PT_BUILD="cpu"
  else
    PT_BUILD="cu90"
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  PT_OS="macos"
  PT_BUILD="cpu"
fi

if [[ "$PT_VERSION" == "latest" ]]; then
  PT_BUILD=nightly/${PT_BUILD}
fi

# Where to get the archive
LIBTORCH_URL=https://download.pytorch.org/libtorch/${PT_BUILD}/libtorch-${PT_OS}-${PT_VERSION}.zip
# Directory where torch is extracted to
LIBTORCH_DIRECTORY=libtorch-${PT_OS}-${PT_VERSION}

# Archive - specifically named
LIBTORCH_ARCHIVE=libtorch-${PT_OS}-${PT_BUILD}-${PT_VERSION}.zip

if [ ! -e "${LIBTORCH_ARCHIVE}" ]; then
  echo "Downloading libtorch ${PT_VERSION} ${PT_BUILD}"
  curl -L ${LIBTORCH_URL} > ${LIBTORCH_ARCHIVE}
fi

unzip -o ${LIBTORCH_ARCHIVE}
tar cf - libtorch | tar xf -  --no-same-owner --strip-components=1 -C ${PREFIX}
rm -rf libtorch

if [[ "${PT_OS}" == "macos" ]]; then
  # also download mkl
  MKL_BUNDLE=mklml_mac_2019.0.1.20180928
  if [ ! -e "${MKL_BUNDLE}.tgz" ]; then
    wget "https://github.com/intel/mkl-dnn/releases/download/v0.17.1/${MKL_BUNDLE}.tgz"
  fi
  tar xf ${MKL_BUNDLE}.tgz --no-same-owner --strip-components=1 -C ${PREFIX}
fi

## ONNXRUNTIME
ORT_VERSION="0.4.0"
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  if [[ "$1" == "cpu" ]]; then
    ORT_OS="linux-x64"
    ORT_BUILD="cpu"
  else
    ORT_OS="linux-x64-gpu"
    ORT_BUILD="gpu"
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ORT_OS="osx-x64"
  ORT_BUILD=""
fi

ORT_ARCHIVE=onnxruntime-${ORT_OS}-${ORT_VERSION}.tgz

if [ ! -e ${ORT_ARCHIVE} ]; then
  echo "Downloading ONNXRuntime ${ORT_VERSION} ${ORT_BUILD}"
  wget https://github.com/Microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}
fi

tar xf ${ORT_ARCHIVE} --no-same-owner --strip-components=1 -C ${PREFIX}

echo "Done"
