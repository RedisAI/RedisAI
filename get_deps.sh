#!/usr/bin/env bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

set -e
[[ $VERBOSE == 1 ]] && set -x

if [[ "$1" == "cpu" ]]; then
  GPU=no
  DEVICE=cpu
elif [[ "$1" == "gpu" ]]; then
  GPU=yes
  DEVICE=gpu
else
  GPU=${GPU:-no}
  if [[ $GPU == 1 ]]; then
    DEVICE=gpu
  else
    DEVICE=cpu
  fi
fi

DEPS_DIR=$HERE/deps
mkdir -p ${DEPS_DIR}
cd ${DEPS_DIR}

PREFIX=${DEPS_DIR}/install
mkdir -p ${PREFIX}

DLPACK_PREFIX=${PREFIX}/dlpack
TF_PREFIX=${PREFIX}/libtensorflow
TORCH_PREFIX=${PREFIX}/libtorch
ORT_PREFIX=${PREFIX}/onnxruntime

## DLPACK

[[ $FORCE == 1 ]] && rm -rf ${DLPACK_PREFIX}

if [[ ! -d dlpack ]]; then
    echo "Cloning dlpack ..."
    git clone --depth 1 https://github.com/dmlc/dlpack.git
    echo "Done."
else
  echo "dlpack is in place."
fi

if [[ ! -d ${DLPACK_PREFIX}/include ]]; then
    mkdir -p ${DLPACK_PREFIX}
    ln -sf ${DEPS_DIR}/dlpack/include ${DLPACK_PREFIX}/include
fi

## TENSORFLOW

TF_VERSION="1.14.0"

[[ $FORCE == 1 ]] && rm -rf ${TF_PREFIX}

if [[ ! -d ${TF_PREFIX} ]]; then
  echo "Installing TensorFlow ..."

  mkdir -p ${TF_PREFIX}

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
    wget -q https://storage.googleapis.com/tensorflow/libtensorflow/${LIBTF_ARCHIVE}
  fi
  
  tar xf ${LIBTF_ARCHIVE} --no-same-owner --strip-components=1 -C ${TF_PREFIX}
  
  echo "Done."
else
  echo "TensorFlow is in place."
fi

## PYTORCH

PT_VERSION="1.2.0"

[[ $FORCE == 1 ]] && rm -rf ${TORCH_PREFIX}

if [[ ! -d ${TORCH_PREFIX} ]]; then
  echo "Installing libtorch ..."

  mkdir -p ${TORCH_PREFIX}

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
    curl -s -L ${LIBTORCH_URL} > ${LIBTORCH_ARCHIVE}
  fi
  
  unzip -q -o ${LIBTORCH_ARCHIVE} -d ${TORCH_PREFIX}/../
  
  echo "Done."
  
  if [[ "${PT_OS}" == "macos" ]]; then
    echo "Installing MKL ..."
    # also download mkl
    MKL_BUNDLE=mklml_mac_2019.0.3.20190220
    if [ ! -e "${MKL_BUNDLE}.tgz" ]; then
      wget -q "https://github.com/intel/mkl-dnn/releases/download/v0.18/${MKL_BUNDLE}.tgz"
    fi
    tar xzf ${MKL_BUNDLE}.tgz --no-same-owner --strip-components=1 -C ${TORCH_PREFIX}
    mkdir -p ${ORT_PREFIX}
    tar xzf ${MKL_BUNDLE}.tgz --no-same-owner --strip-components=1 -C ${ORT_PREFIX}
    echo "Done."
  fi
else
  echo "librotch is in place."
fi

## ONNXRUNTIME
ORT_VERSION="0.5.0"
ORT_ARCH="x64"
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  if [[ "$1" == "cpu" ]]; then
    ORT_OS="linux"
    ORT_BUILD=""
  else
    ORT_OS="linux"
    ORT_BUILD="-gpu"
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ORT_OS="osx"
  ORT_BUILD=""
fi

[[ $FORCE == 1 ]] && rm -rf ${ORT_PREFIX}

ORT_ARCHIVE=onnxruntime-${ORT_OS}-${ORT_ARCH}${ORT_BUILD}-${ORT_VERSION}.tgz

if [[ ! -d ${ORT_PREFIX} ]]; then
  echo "Installing ONNXRuntime ..."

  mkdir -p ${ORT_PREFIX}

  if [ ! -e ${ORT_ARCHIVE} ]; then
    echo "Downloading ONNXRuntime ${ORT_VERSION} ${DEVICE}"
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}
  fi
  
  tar xf ${ORT_ARCHIVE} --no-same-owner --strip-components=1 -C ${ORT_PREFIX}
else
  echo "onnxruntime is in place."
fi

echo "Done."
