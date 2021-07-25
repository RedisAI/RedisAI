#!/bin/bash

set -e
VER="$1"  # 1.8.0
PLATFORM="$2"  # x64|jetson
BUILDTYPE="$3"  # Release
BASEOS="$4"  # linux (mac future?)
VARIANT="$5"  # if set (gpu)

if [ ${BASEOS} == "MacOS" ]; then
BASEDIR=`pwd`  # already in the directory on a mac
PACKDIR=${BASEDIR}/pack/
target_os=osx
else
BASEDIR=`pwd`/onnxruntime
PACKDIR=${BASEDIR}/pack/
target_os=${BASEOS,,}
fi

target=onnxruntime-${target_os}-${PLATFORM}-${VER}

if [ ! -z "${VARIANT}" ]; then
    target=onnxruntime-${target_os}-${PLATFORM}-${VARIANT}-${VER}
fi

mkdir -p ${PACKDIR}/include ${PACKDIR}/lib
cp ${BASEDIR}/docs/C_API_Guidelines.md ${PACKDIR}
cp ${BASEDIR}/LICENSE ${PACKDIR}
cp ${BASEDIR}/README.md ${PACKDIR}
cp ${BASEDIR}/ThirdPartyNotices.txt ${PACKDIR}
cp ${BASEDIR}/VERSION_NUMBER ${PACKDIR}
cp ${BASEDIR}/include/onnxruntime/core/session/onnxruntime_c_api.h ${PACKDIR}/include/
cp ${BASEDIR}/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h ${PACKDIR}/include/

if [ ${BASEOS} == "MacOS" ]; then
cp ${BASEDIR}/build/${BASEOS}/$BUILDTYPE/libonnxruntime.${VER}.dylib ${PACKDIR}/lib/
else
cp ${BASEDIR}/build/${BASEOS}/$BUILDTYPE/libonnxruntime.so.${VER} ${PACKDIR}/lib/
fi

# hash
cd ${BASEDIR}/
git rev-parse HEAD > ${PACKDIR}/GIT_COMMIT_ID

cd pack/lib/
if [ ${BASEOS} != "MacOS" ]; then
ln -s libonnxruntime.so.${VER} libonnxruntime.so
else
ln -s libonnxruntime.${VER}.dylib libonnxruntime.dylib
fi

cd ${PACKDIR}/..
mv pack ${target}
tar czf ${target}.tgz ${target}/
ls -l *.tgz
