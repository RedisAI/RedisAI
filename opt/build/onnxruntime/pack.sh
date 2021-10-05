#!/bin/bash

set -e
VER="$1"  # 1.8.0
PLATFORM="$2"  # x64|jetson
BUILDTYPE="$3"  # Release
BASEOS="$4"  # linux (mac future?)
VARIANT="$5"  # if set (gpu)

target=onnxruntime-${BASEOS}-${PLATFORM}-${VER}
if [ ! -z "${VARIANT}" ]; then
    target=onnxruntime-${BASEOS}-${PLATFORM}-${VARIANT}-${VER}
fi

mkdir -p pack/include pack/lib
cp onnxruntime/build/Linux/$BUILDTYPE/libonnxruntime.so.${VER} pack/lib/
cp onnxruntime/docs/C_API_Guidelines.md pack/
cp onnxruntime/LICENSE pack/
cp onnxruntime/README.md pack/
cp onnxruntime/ThirdPartyNotices.txt pack/
cp onnxruntime/VERSION_NUMBER pack/
cd onnxruntime/
git rev-parse HEAD > ../pack/GIT_COMMIT_ID
cd ..
cp onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h pack/include/
cp onnxruntime/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h pack/include/
cd pack/lib/
ln -s libonnxruntime.so.${VER} libonnxruntime.so
cd ../..
mv pack ${target}
tar czf ${target}.tgz ${target}/
