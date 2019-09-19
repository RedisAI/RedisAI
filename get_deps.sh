#!/usr/bin/env bash

error() {
	echo "There are errors."
	exit 1
}

trap error ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ $1 == --help || $1 == help ]]; then
	cat <<-END
		get_deps.sh [cpu|gpu] [--help|help]
		
		Argument variables:
		VERBOSE=1   Print commands
		FORCE=1     Download even if present
		WITH_TF=0   Skip Tensorflow
		WITH_PT=0   Skip PyTorch
		WITH_ORT=0  Skip OnnxRuntime

	END
	exit 0
fi

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

OS=$(python3 $HERE/opt/readies/bin/platform --os)
ARCH=$(python3 $HERE/opt/readies/bin/platform --arch)

# avoid wget warnings on macOS
[[ $OS == macosx ]] && export LC_ALL=en_US.UTF-8

DEPS_DIR=$HERE/deps/$OS-$ARCH-$DEVICE
mkdir -p ${DEPS_DIR}
cd ${DEPS_DIR}

# PREFIX=${DEPS_DIR}/install
# mkdir -p ${PREFIX}

DLPACK=dlpack
LIBTENSORFLOW=libtensorflow
LIBTORCH=libtorch
MKL=mkl
ONNXRUNTIME=onnxruntime

######################################################################################## DLPACK

[[ $FORCE == 1 ]] && rm -rf $DLPACK

if [[ ! -d $DLPACK ]]; then
	echo "Cloning dlpack ..."
    git clone --depth 1 https://github.com/dmlc/dlpack.git $DLPACK
	echo "Done."
else
	echo "dlpack is in place."
fi

################################################################################# LIBTENSORFLOW

TF_VERSION="1.14.0"

if [[ $WITH_TF != 0 ]]; then
	[[ $FORCE == 1 ]] && rm -rf $LIBTENSORFLOW

	if [[ ! -d $LIBTENSORFLOW ]]; then
		echo "Installing TensorFlow ..."
		
		if [[ $OS == linux ]]; then
			TF_OS="linux"
			if [[ $GPU == no ]]; then
				TF_BUILD="cpu"
			else
				TF_BUILD="gpu"
			fi
			if [[ $ARCH == x64 ]]; then
				TF_VERSION=1.14.0
				TF_ARCH=x86_64
				LIBTF_URL_BASE=https://storage.googleapis.com/tensorflow/libtensorflow
			elif [[ $ARCH == arm64v8 ]]; then
				TF_VERSION=1.14.0
				TF_ARCH=arm64
				LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
			elif [[ $ARCH == arm32v7 ]]; then
				TF_VERSION=1.14.0
				TF_ARCH=arm
				LIBTF_URL_BASE=https://s3.amazonaws.com/redismodules/tensorflow
			fi
		elif [[ $OS == macosx ]]; then
			TF_VERSION=1.14.0
			TF_OS=darwin
			TF_BUILD=cpu
			TF_ARCH=x86_64
			LIBTF_URL_BASE=https://storage.googleapis.com/tensorflow/libtensorflow
		fi

		LIBTF_ARCHIVE=libtensorflow-${TF_BUILD}-${TF_OS}-${TF_ARCH}-${TF_VERSION}.tar.gz

		[[ ! -f $LIBTF_ARCHIVE || $FORCE == 1 ]] && wget --quiet $LIBTF_URL_BASE/$LIBTF_ARCHIVE

		rm -rf $LIBTENSORFLOW.x
		mkdir $LIBTENSORFLOW.x
		tar xf $LIBTF_ARCHIVE --no-same-owner -C $LIBTENSORFLOW.x
		mv $LIBTENSORFLOW.x $LIBTENSORFLOW
		
		echo "Done."
	else
		echo "TensorFlow is in place."
	fi
else
	echo "Skipping TensorFlow."
fi # WITH_TF

###################################################################################### LIBTORCH

PT_VERSION="1.2.0"

if [[ $WITH_PT != 0 ]]; then
	[[ $FORCE == 1 ]] && rm -rf $LIBTORCH

	if [[ ! -d $LIBTORCH ]]; then
		echo "Installing libtorch ..."

		PT_REPACK=0
		
		if [[ $OS == linux ]]; then
			PT_OS=linux
			if [[ $GPU == no ]]; then
				PT_BUILD=cpu
			else
				PT_BUILD=cu100
			fi
			if [[ $ARCH == x64 ]]; then
				PT_ARCH=x86_64
				PT_REPACK=1
			elif [[ $ARCH == arm64v8 ]]; then
				PT_ARCH=arm64
			elif [[ $ARCH == arm32v7 ]]; then
				PT_ARCH=arm
			fi
		elif [[ $OS == macosx ]]; then
			PT_OS=macos
			PT_ARCH=x86_64
			PT_BUILD=cpu
			PT_REPACK=1
		fi

		[[ $PT_VERSION == latest ]] && PT_BUILD=nightly/${PT_BUILD}

		LIBTORCH_ARCHIVE=libtorch-${PT_BUILD}-${PT_OS}-${PT_ARCH}-${PT_VERSION}.tar.gz

		if [[ $PT_REPACK == 1 ]]; then
			PT_VERSION=$PT_VERSION $HERE/opt/build/libtorch/repack.sh
		else
			LIBTORCH_URL=https://s3.amazonaws.com/redismodules/pytorch/$LIBTORCH_ARCHIVE

			[[ ! -f $LIBTORCH_ARCHIVE || $FORCE == 1 ]] && wget -q $LIBTORCH_URL
		fi
		
		rm -rf $LIBTORCH.x
		mkdir $LIBTORCH.x

		tar xf $LIBTORCH_ARCHIVE --no-same-owner -C $LIBTORCH.x
		mv $LIBTORCH.x/libtorch $LIBTORCH
		rmdir $LIBTORCH.x
		
		echo "Done."
	else
		echo "librotch is in place."
	fi
else
	echo "SKipping libtorch."
fi # WITH_PT

########################################################################################### MKL

if [[ ! -d mkl ]]; then
	MKL_VERSION=0.18
	MKL_BUNDLE_VER=2019.0.3.20190220
	if [[ $OS == macosx ]]; then
		echo "Installing MKL ..."

		MKL_OS=mac
		MKL_ARCHIVE=mklml_${MKL_OS}_${MKL_BUNDLE_VER}.tgz
		[[ ! -e ${MKL_ARCHIVE} ]] && wget -q https://github.com/intel/mkl-dnn/releases/download/v${MKL_VERSION}/${MKL_ARCHIVE}

		rm -rf $MKL.x
		mkdir $MKL.x
		tar xzf ${MKL_ARCHIVE} --no-same-owner --strip-components=1 -C $MKL.x
		mv $MKL.x $MKL

		echo "Done."
	fi
else
	echo "mkl is in place."
fi

################################################################################### ONNXRUNTIME

ORT_VERSION="0.5.0"

if [[ $WITH_ORT != 0 ]]; then
	[[ $FORCE == 1 ]] && rm -rf $ONNXRUNTIME

	if [[ ! -d $ONNXRUNTIME ]]; then
		echo "Installing ONNXRuntime ..."

		if [[ $OS == linux ]]; then
			ORT_OS=linux
			if [[ $GPU == no ]]; then
				ORT_BUILD=""
			else
				ORT_BUILD="-gpu"
			fi
			if [[ $ARCH == x64 ]]; then
				ORT_ARCH=x64
				ORT_URL_BASE=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}
			elif [[ $ARCH == arm64v8 ]]; then
				ORT_ARCH=arm64
				ORT_URL_BASE=https://s3.amazonaws.com/redismodules/onnxruntime
			elif [[ $ARCH == arm32v7 ]]; then
				ORT_ARCH=arm
				ORT_URL_BASE=https://s3.amazonaws.com/redismodules/onnxruntime
			fi
		elif [[ $OS == macosx ]]; then
			ORT_OS=osx
			ORT_ARCH=x64
			ORT_BUILD=""
			ORT_URL_BASE=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}
		fi

		ORT_ARCHIVE=onnxruntime-${ORT_OS}-${ORT_ARCH}${ORT_BUILD}-${ORT_VERSION}.tgz

		[[ ! -e ${ORT_ARCHIVE} ]] && wget -q $ORT_URL_BASE/${ORT_ARCHIVE}

		rm -rf $ONNXRUNTIME.x
		mkdir $ONNXRUNTIME.x
		tar xzf ${ORT_ARCHIVE} --no-same-owner --strip-components=1 -C $ONNXRUNTIME.x
		mv $ONNXRUNTIME.x $ONNXRUNTIME
		
		echo "Done."
	else
		echo "ONNXRuntime is in place."
	fi
else
	echo "Skipping ONNXRuntime."
fi # WITH_ORT
