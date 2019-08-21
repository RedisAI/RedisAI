#!/bin/bash

set -e
VER="$1"

mkdir -p pack/include pack/lib
cp onnxruntime/build/Linux/MinSizeRel/libonnxruntime.so.${VER} pack/lib/
cp onnxruntime/docs/C_API.md pack/
cp onnxruntime/LICENSE pack/
cp onnxruntime/README.md pack/
cp onnxruntime/ThirdPartyNotices.txt pack/
cp onnxruntime/VERSION_NUMBER pack/
cd onnxruntime/
git rev-parse HEAD > ../pack/GIT_COMMIT_ID
cd ..
cp onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h pack/include/
cd pack/lib/
ln -s libonnxruntime.so.${VER} libonnxruntime.so
cd ../..
mv pack onnxruntime-linux-arm64-${VER}
tar czf onnxruntime-linux-arm64-${VER}.tgz onnxruntime-linux-arm64-${VER}/
