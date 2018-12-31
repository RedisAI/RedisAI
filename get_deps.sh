#!/usr/bin/env bash

BASE_DIRECTORY=`pwd`

DEPS_DIRECTORY=${BASE_DIRECTORY}/deps

mkdir -p ${DEPS_DIRECTORY}

## DLPACK

cd ${DEPS_DIRECTORY}

DLPACK_DIRECTORY=${DEPS_DIRECTORY}/dlpack

echo "Cloning dlpack"
git clone --depth 1 https://github.com/dmlc/dlpack.git

## REDIS

cd ${DEPS_DIRECTORY}

echo "Building Redis"
git clone --depth 1 --branch 5.0 https://github.com/antirez/redis.git

cd ${DEPS_DIRECTORY}/redis

make MALLOC=libc

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

LIBTF_DIRECTORY=`pwd`/libtensorflow

if [ ! -d "${LIBTF_DIRECTORY}" ]; then
  echo "Downloading libtensorflow ${TF_VERSION} ${TF_BUILD}"
  mkdir -p ${LIBTF_DIRECTORY}
  curl -L \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_BUILD}-${TF_OS}-x86_64-${TF_VERSION}.tar.gz" |
    tar -C ${LIBTF_DIRECTORY} -xz
fi

## PYTORCH

cd ${DEPS_DIRECTORY}

PT_VERSION="1.0.0"
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
  curl -L \
    "https://download.pytorch.org/libtorch/${PT_BUILD}/libtorch-${PT_OS}-${PT_VERSION}.zip" |
    tar -C . -xz
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

## UTIL

cd ${DEPS_DIRECTORY}

LIBTORCH_C_SRC_DIRECTORY=${BASE_DIRECTORY}/util/libtorch_c
LIBTORCH_C_DIRECTORY=${DEPS_DIRECTORY}/libtorch_c

mkdir -p ${LIBTORCH_C_DIRECTORY}/build
cd ${LIBTORCH_C_DIRECTORY}/build

cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_DIRECTORY} \
      -DDLPACK_DIRECTORY=${DLPACK_DIRECTORY} \
      -DCMAKE_INSTALL_PREFIX=${LIBTORCH_C_DIRECTORY} \
      ${LIBTORCH_C_SRC_DIRECTORY}
make -j2 && make install

cd ${LIBTORCH_C_DIRECTORY}
rm -rf ${LIBTORCH_C_DIRECTORY}/build

## TEST

cd ${BASE_DIRECTORY}/test

echo "Testing TensorFlow"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIBTF_DIRECTORY}/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${LIBTF_DIRECTORY}/lib
fi
gcc -I${LIBTF_DIRECTORY}/include \
    -L${LIBTF_DIRECTORY}/lib \
    tf_api_test.c -ltensorflow && \
    ./a.out && rm a.out

echo "Testing PyTorch"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIBTORCH_DIRECTORY}/lib:${LIBTORCH_C_DIRECTORY}/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${LIBTORCH_DIRECTORY}/lib:${LIBTORCH_C_DIRECTORY}/lib
fi
gcc -I${LIBTORCH_C_DIRECTORY}/include \
    -I${DLPACK_DIRECTORY}/include \
    -L${LIBTORCH_C_DIRECTORY}/lib \
    pt_api_test.c -ltorch_c && \
    ./a.out && \
    rm a.out

## DONE

cd ${BASE_DIRECTORY}

echo "Done"

