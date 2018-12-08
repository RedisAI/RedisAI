#!/usr/bin/env bash

mkdir -p deps
cd deps

echo "Cloning dlpack"
git clone https://github.com/dmlc/dlpack.git

echo "Building Redis"
git clone --branch 5.0 https://github.com/antirez/redis.git
cd redis
make MALLOC=libc
cd ..

TF_VERSION="1.12.0"
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  OS="linux"
  if [[ "$1" == "cpu" ]]; then
    TF_TYPE="cpu"
  else
    TF_TYPE="gpu"
  fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS="darwin"
  TF_TYPE="cpu"
fi

echo "Downloading libtensorflow ${TF_VERSION} ${TF_TYPE}"

LIBTF_DIRECTORY=`pwd`/libtensorflow
mkdir -p $LIBTF_DIRECTORY
curl -L \
  "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-${TF_VERSION}.tar.gz" |
  tar -C $LIBTF_DIRECTORY -xz

cd ..

cd test

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTF_DIRECTORY/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LIBTF_DIRECTORY/lib
fi
gcc -I$LIBTF_DIRECTORY/include -L$LIBTF_DIRECTORY/lib tf_api_test.c -ltensorflow && ./a.out && rm a.out

cd ..

echo "Done"

