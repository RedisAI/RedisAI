#!/bin/bash

error() {
	>&2 echo "$0: There are errors."
	exit 1
}

if [[ -z $_Dbg_DEBUGGER_LEVEL ]]; then
	trap error ERR
fi

#----------------------------------------------------------------------------------------------

if [[ $1 == --help || $1 == help ]]; then
	cat <<-END
		pack.sh [cpu|gpu] [--help|help]
		
		Argument variables:
		DEVICE=cpu|gpu    CPU or GPU variants
		RAMP=1            Build RAMP file
		DEPS=1            Build dependencies file

		VARIANT=name      Build variant (empty for standard packages)
		BRANCH=branch     Branch names to serve as an exta package tag
		GITSHA=1          Append Git SHA to shapshot package names

		BINDIR=dir        Directory in which packages are created
		INSTALL_DIR=dir   Directory in which artifacts are found

	END
	exit 0
fi

#----------------------------------------------------------------------------------------------

[[ $IGNERR == 1 ]] || set -e
[[ $VERBOSE == 1 ]] && set -x

[[ -z $DEVICE ]] && { echo DEVICE undefined; exit 1; }
[[ -z $BINDIR ]] && { echo BINDIR undefined; exit 1; }
[[ -z $INSTALL_DIR ]] && { echo INSTALL_DIR undefined; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $HERE/readies/shibumi/functions
ROOT=$(realpath $HERE/..)
READIES=$ROOT/opt/readies/bin

RAMP_PROG="python3 -m RAMP.ramp"

BINDIR=$(realpath $BINDIR)
INSTALL_DIR=$(realpath $INSTALL_DIR)

. $READIES/enable-utf8

export ARCH=$($READIES/platform --arch)
export OS=$($READIES/platform --os)
export OSNICK=$($READIES/platform --osnick)

export PRODUCT=redisai
export PRODUCT_LIB=$PRODUCT.so

export PACKAGE_NAME=${PACKAGE_NAME:-${PRODUCT}}

# BACKENDS="all torch tensorflow onnxruntime tflite"
BACKENDS="torch tensorflow onnxruntime tflite"

#----------------------------------------------------------------------------------------------

pack_ramp() {
	cd $ROOT

	local platform="$OS-$OSNICK-$ARCH"
	local stem=${PACKAGE_NAME}-${DEVICE}.${platform}

	if [[ $SNAPSHOT == 0 ]]; then
		local verspec=${SEMVER}${VARIANT}
		local packdir=.
		local s3base=""
	else
		local verspec=${BRANCH}${VARIANT}
		local packdir=snapshots
		local s3base=snapshots/
	fi
	
	local fq_package=$stem.${verspec}.zip

	[[ ! -d $BINDIR/$packdir ]] && mkdir -p $BINDIR/$packdir

	local packdir="$BINDIR/$packdir"
	local packfile="$packdir/$fq_package"
	local product_so="$INSTALL_DIR/$PRODUCT.so"

	local xtx_vars=""
	for dep in $BACKENDS; do
		eval "export NAME_${dep}=${PACKAGE_NAME}_${dep}"
		local dep_fname=${PACKAGE_NAME}-${DEVICE}-${dep}.${platform}.${verspec}.tgz
		eval "export PATH_${dep}=${s3base}${dep_fname}"
		local dep_sha256="$packdir/${dep_fname}.sha256"
		eval "export SHA256_${dep}=$(cat $dep_sha256)"

		xtx_vars+=" -e NAME_$dep -e PATH_$dep -e SHA256_$dep"
	done
	
	python3 $READIES/xtx \
		$xtx_vars \
		-e DEVICE -e NUMVER -e SEMVER \
		$ROOT/ramp.yml > /tmp/ramp.yml
	rm -f /tmp/ramp.fname $packfile
	$RAMP_PROG pack -m /tmp/ramp.yml --packname-file /tmp/ramp.fname --verbose --debug -o $packfile $product_so >/tmp/ramp.err 2>&1 || true
	if [[ ! -e $packfile ]]; then
		>&2 echo "Error generating RAMP file:"
		>&2 cat /tmp/ramp.err
		exit 1
	fi
}

#----------------------------------------------------------------------------------------------

pack_deps() {
	local depname="$1"
	
	cd $ROOT

	local platform="$OS-$OSNICK-$ARCH"
	local stem=${PACKAGE_NAME}-${DEVICE}-${depname}.${platform}
	local fq_package=$stem.${SEMVER}${VARIANT}.tgz
	local tar_path=$BINDIR/$fq_package
	local backends_prefix_dir=""
	
	if [[ $depname == all ]]; then
		local backends_dir=.
	else
		local backends_dir=${PRODUCT}_$depname
	fi
	
	cd $INSTALL_DIR/backends
	{ find $backends_dir -name "*.so*" | \
	  xargs tar -c --sort=name --owner=root:0 --group=root:0 --mtime='UTC 1970-01-01' --transform "s,^,$backends_prefix_dir," 2>> /tmp/pack.err | \
	  gzip -n - > $tar_path ; E=$?; } || true
	sha256sum $tar_path | gawk '{print $1}' > $tar_path.sha256

	mkdir -p $BINDIR/snapshots
	cd $BINDIR/snapshots
	if [[ ! -z $BRANCH ]]; then
		local snap_package=$stem.${BRANCH}${VARIANT}.tgz
		ln -sf ../$fq_package $snap_package
		ln -sf ../$fq_package.sha256 $snap_package.sha256
	fi
}

#----------------------------------------------------------------------------------------------

export NUMVER=$(NUMERIC=1 $ROOT/opt/getver)
export SEMVER=$($ROOT/opt/getver)

if [[ ! -z $VARIANT ]]; then
	VARIANT=-${VARIANT}
fi

[[ -z $BRANCH ]] && BRANCH=${CIRCLE_BRANCH:-`git rev-parse --abbrev-ref HEAD`}
BRANCH=${BRANCH//[^A-Za-z0-9._-]/_}
if [[ $GITSHA == 1 ]]; then
	GIT_COMMIT=$(git describe --always --abbrev=7 --dirty="+" 2>/dev/null || git rev-parse --short HEAD)
	BRANCH="${BRANCH}-${GIT_COMMIT}"
fi
export BRANCH

[[ $DEPS == 1 ]] && echo "Building dependencies ..."
for dep in $BACKENDS; do
	if [[ $DEPS == 1 ]]; then
		echo "$dep ..."
		pack_deps $dep
	fi
done

if [[ $RAMP == 1 ]]; then
	if ! command -v redis-server > /dev/null; then
		>&2 echo "$0: Cannot find redis-server. Aborting."
		exit 1
	fi

	echo "Building RAMP files ..."
	SNAPSHOT=0 pack_ramp
	SNAPSHOT=1 pack_ramp
	echo "Done."
fi

exit 0
