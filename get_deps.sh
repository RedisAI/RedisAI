#!/usr/bin/env bash
set -e
set -x

BASE_DIRECTORY=`pwd`

# Allow different deps for different platforms:
PLATNAME=${OSTYPE}
if [ -e /etc/debian-version ]; then
    PLATNAME=${PLATNAME}-deb
fi

DEPS_DIRECTORY=${BASE_DIRECTORY}/deps-${PLATNAME}

mkdir -p ${DEPS_DIRECTORY}

## DLPACK

cd ${DEPS_DIRECTORY}

DLPACK_DIRECTORY=${DEPS_DIRECTORY}/dlpack

if [ ! -d dlpack ]; then
    echo "Cloning dlpack"
    git clone --depth 1 https://github.com/dmlc/dlpack.git
fi

## REDIS

cd ${DEPS_DIRECTORY}

if [ ! -d redis ]; then
    echo "Building Redis"
    git clone --depth 1 --branch 5.0 https://github.com/antirez/redis.git
    cd ${DEPS_DIRECTORY}/redis
    make MALLOC=libc
fi

## TENSORFLOW

cd ${DEPS_DIRECTORY}

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

LIBTF_DIRECTORY=${DEPS_DIRECTORY}/libtensorflow

if [ ! -d "${LIBTF_DIRECTORY}" ]; then
  # rm -rf ${LIBTF_DIRECTORY}
  echo "Downloading libtensorflow ${TF_VERSION} ${TF_BUILD}"
  mkdir -p ${LIBTF_DIRECTORY}
  curl -L \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_BUILD}-${TF_OS}-x86_64-${TF_VERSION}.tar.gz" |
    tar -C ${LIBTF_DIRECTORY} -xz
fi

## PYTORCH

cd ${DEPS_DIRECTORY}

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

LIBTORCH_DIRECTORY=`pwd`/libtorch

if [ ! -d "${LIBTORCH_DIRECTORY}" ]; then
  echo "Downloading libtorch ${PT_VERSION} ${PT_BUILD}"
  # rm -rf ${LIBTORCH_DIRECTORY}
  curl -o tmp.zip \
    "https://download.pytorch.org/libtorch/${PT_BUILD}/libtorch-${PT_OS}-${PT_VERSION}.zip" && \
    unzip tmp.zip && rm tmp.zip
  if [[ "$PT_OS" == "macos" ]]; then
    mkdir -p ./mkl_tmp
    MKL_BUNDLE=mklml_mac_2019.0.1.20180928
    curl -L \
      "https://github.com/intel/mkl-dnn/releases/download/v0.17.1/${MKL_BUNDLE}.tgz" |
      tar -C ./mkl_tmp -xz
    mv mkl_tmp/${MKL_BUNDLE}/license.txt ${LIBTORCH_DIRECTORY}/mkl_license.txt
    mv mkl_tmp/${MKL_BUNDLE}/lib/* ${LIBTORCH_DIRECTORY}/lib
    mkdir -p ${LIBTORCH_DIRECTORY}/include/mkl
    mv mkl_tmp/${MKL_BUNDLE}/include/* ${LIBTORCH_DIRECTORY}/include/mkl
    rm -rf mkl_tmp
  fi
fi

## INSTALL

DEPS_INSTALL_LIB_DIRECTORY=${DEPS_DIRECTORY}/install/lib
mkdir -p ${DEPS_INSTALL_LIB_DIRECTORY}

cp -R ${DEPS_DIRECTORY}/libtensorflow/lib/* ${DEPS_INSTALL_LIB_DIRECTORY}
cp -R ${DEPS_DIRECTORY}/libtorch/lib/* ${DEPS_INSTALL_LIB_DIRECTORY}
cp -R ${DEPS_DIRECTORY}/libtorch_c/lib/* ${DEPS_INSTALL_LIB_DIRECTORY}

## DONE

cd ${BASE_DIRECTORY}

echo "Done"
