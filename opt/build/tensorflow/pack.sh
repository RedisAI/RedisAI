#!/bin/bash

VERSION=$1 # 2.5.0
BASEOS=$2  # linux
VARIANT=$3 # cpu or gpu
if [ ! -z "$4" ]; then
    PLATFORM=$4 # x86_64|jetson
else
    PLATFORM=`uname -m`
fi

target=libtensorflow-${VARIANT}-${BASEOS}-${PLATFORM}-${VERSION}

mkdir -p pack/include/tensorflow pack/lib
rsync -aqH --recursive tensorflow/tensorflow/c --include '*/' --include '*.h' --exclude '*'  pack/include/tensorflow
rsync -aqH --recursive tensorflow/tensorflow/core --include '*/' --include '*.h' --exclude '*'  pack/include/tensorflow
cp tensorflow/LICENSE pack
cp tensorflow/bazel-bin/tensorflow/libtensorflow.so pack/lib
cp tensorflow/bazel-bin/tensorflow/*so* pack/lib
mv pack ${target}
tar czf ${target}.tar.gz ${target}/
