#!/usr/bin/env bash

mkdir -p deps
cd deps

echo "Building Redis"
git clone --branch 5.0 https://github.com/antirez/redis.git
cd redis
make MALLOC=libc
cd ..

echo "Downloading libtensorflow"
TF_TYPE="cpu" # Change to "gpu" for GPU support
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS="darwin"
fi
LIBTF_DIRECTORY=`pwd`/libtensorflow
mkdir -p $LIBTF_DIRECTORY
curl -L \
  "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.10.1.tar.gz" |
  tar -C $LIBTF_DIRECTORY -xz

cd ..

cd test

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTF_DIRECTORY/lib
fi
gcc -I$LIBTF_DIRECTORY/include -L$LIBTF_DIRECTORY/lib tf_api_test.c -ltensorflow && ./a.out && rm a.out

cd ..

echo "Done"

