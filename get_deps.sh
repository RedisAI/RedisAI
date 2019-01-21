#!/usr/bin/env bash

mkdir -p deps
cd deps

## DLPACK

echo "Cloning dlpack"
git clone --depth 1 https://github.com/dmlc/dlpack.git


echo "Building Redis"
git clone --depth 1 --branch 5.0 https://github.com/antirez/redis.git
cd redis
make MALLOC=libc
cd ..

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

LIBTF_DIRECTORY=`pwd`/libtensorflow

if [ ! -d "$LIBTF_DIRECTORY" ]; then
  echo "Downloading libtensorflow ${TF_VERSION} ${TF_BUILD}"
  mkdir -p $LIBTF_DIRECTORY
  curl -L \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_BUILD}-${TF_OS}-x86_64-${TF_VERSION}.tar.gz" |
    tar -C $LIBTF_DIRECTORY -xz
fi

## PYTORCH

#PT_VERSION="1.0.0"
PT_VERSION="latest"

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

LIBPT_DIRECTORY=`pwd`/libtorch

#if [ ! -d "$LIBPT_DIRECTORY" ]; then
#  echo "Downloading libtorch ${PT_VERSION} ${PT_BUILD}"
#  curl -L \
#    "https://download.pytorch.org/libtorch/${PT_BUILD}/libtorch-${PT_OS}-${PT_VERSION}.zip" |
#    tar -C . -xz
#fi

cd ..

## TEST

cd test

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTF_DIRECTORY/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LIBTF_DIRECTORY/lib
fi
gcc -I$LIBTF_DIRECTORY/include -L$LIBTF_DIRECTORY/lib tf_api_test.c -ltensorflow && ./a.out && rm a.out

cd ..

echo "Done"

