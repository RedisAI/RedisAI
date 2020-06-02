#!/bin/bash

[[ $IGNERR == 1 ]] || set -e
[[ $VERBOSE == 1 ]] && set -x

[[ -z $DEVICE ]] && { echo DEVICE undefined; exit 1; }
[[ -z $BINDIR ]] && { echo BINDIR undefined; exit 1; }
[[ -z $INSTALL_DIR ]] && { echo INSTALL_DIR undefined; exit 1; }
[[ ! -z $INTO ]] && INTO=$(realpath $INTO)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $HERE/readies/shibumi/functions
ROOT=$(realpath $HERE/..)

RAMP_PROG="python3 -m RAMP.ramp"
REDIS_ENT_LIB_PATH=/opt/redislabs/lib

BINDIR=$(realpath $BINDIR)
INSTALL_DIR=$(realpath $INSTALL_DIR)

. $ROOT/opt/readies/bin/enable-utf8

export ARCH=$($ROOT/opt/readies/bin/platform --arch)
export OS=$($ROOT/opt/readies/bin/platform --os)
export OSNICK=$($ROOT/opt/readies/bin/platform --osnick)

pack_ramp() {
	echo "Building RAMP file ..."
	cd $ROOT
	
	local STEM=$PRODUCT.$OS-$OSNICK-$ARCH
	local FQ_VER=$GIT_VER
	local FQ_PACKAGE=$STEM.$FQ_VER

	# this is only to extract {semantic_version} into VERSION
	RAMPOUT=$(mktemp /tmp/ramp.XXXXXX)
	$RAMP_PROG pack -m $ROOT/ramp.yml --packname-file $RAMPOUT -o $BINDIR/$PRODUCT.{os}-{architecture}.{semantic_version}.zip $INSTALL_DIR/$PRODUCT.so
	local rampfile=`realpath $(tail -1 $RAMPOUT)`
	rm -f $rampfile $RAMPOUT
	echo `basename $rampfile` | sed -e "s/[^.]*\.[^.]*\.\(.*\)\.zip/\1/" > $BINDIR/VERSION
	export VERSION=$(cat $BINDIR/VERSION)

	$RAMP_PROG pack -m $ROOT/ramp.yml -o $BINDIR/$FQ_PACKAGE.zip \
		-c "BACKENDSPATH $REDIS_ENT_LIB_PATH/$PRODUCT-$DEVICE-$VERSION/backends" $INSTALL_DIR/$PRODUCT.so > /dev/null 2>&1

	cd "$BINDIR"
	ln -sf $FQ_PACKAGE.zip $STEM.$VERSION.zip
	ln -sf $FQ_PACKAGE.zip $STEM.latest.zip
	# [[ ! -z $BRANCH ]] && ln -sf $FQ_PACKAGE.zip $STEM.$BRANCH.zip

	export RELEASE_ARTIFACTS="$RELEASE_ARTIFACTS $STEM.$VERSION.zip $RAMP_STEM.latest.zip"
	export DEV_ARTIFACTS="$DEV_ARTIFACTS $FQ_PACKAGE.zip $STEM.$BRANCH.zip"
	# [[ ! -z $BRANCH ]] && export DEV_ARTIFACTS="$DEV_ARTIFACTS $DEPS.$BRANCH.tgz"
	
	echo "Done."
}

pack_deps() {
	echo "Building dependencies file ..."

	cd $ROOT

	VERSION=$(cat $BINDIR/VERSION)
	local VARIANT=$OS-$OSNICK-$ARCH.$GIT_VER

	local STEM=$PRODUCT-$DEVICE-dependencies.$OS-$OSNICK-$ARCH
	local FQ_VER=$GIT_VER
	local FQ_PACKAGE=$STEM.$FQ_VER
	
	cd $INSTALL_DIR
	local BACKENDS_DIR=$PRODUCT-$DEVICE-$VERSION
	if [[ ! -h backends ]]; then
		[[ ! -d backends ]] && { echo "install-$DEVICE/backend directory not found." ; exit 1; }
		rm -rf $BACKENDS_DIR
		mkdir $BACKENDS_DIR
	
		mv backends $BACKENDS_DIR
		ln -s $BACKENDS_DIR/backends backends
	fi
	find $BACKENDS_DIR -name "*.so*" | xargs tar pczf $BINDIR/$FQ_PACKAGE.tgz
	
	cd "$BINDIR"
	ln -sf $FQ_PACKAGE.tgz $STEM.$VERSION.tgz
	ln -sf $FQ_PACKAGE.tgz $STEM.latest.tgz
	# [[ ! -z $BRANCH ]] && ln -sf $FQ_PACKAGE.tgz $STEM.$BRANCH.tgz
	
	export RELEASE_ARTIFACTS="$RELEASE_ARTIFACTS $STEM.$VERSION.tgz $STEM.latest.tgz"
	export DEV_ARTIFACTS="$DEV_ARTIFACTS $FQ_PACKAGE.tgz"
	# [[ ! -z $BRANCH ]] && export DEV_ARTIFACTS="$DEV_ARTIFACTS $STEM.$BRANCH.tgz"

	echo "Done."
}

if [[ $1 == --help || $1 == help ]]; then
	cat <<-END
		pack.sh [cpu|gpu] [--help|help]
		
		Argument variables:
		BINDIR=dir        directory in which packages are created
		INSTALL_DIR=dir   directory in which artifacts are found
		DEVICE=cpu|gpu
		BRANCH=branch     branch names to serve as an exta package tag
		INTO=dir          package destination directory (optinal)
		RAMP=1            build RAMP file
		DEPS=1            build dependencies file
		RAMP_PROG         path to RAMP program

	END
	exit 0
fi

PRODUCT=redisai
PRODUCT_LIB=$PRODUCT.so

GIT_VER=""
if [[ -d $ROOT/.git ]]; then
	if [[ ! -z $BRANCH ]]; then
		GIT_BRANCH="$BRANCH"
	else
		GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
	fi
	GIT_COMMIT=$(git describe --always --abbrev=7 --dirty="+")
	GIT_VER="${GIT_BRANCH}-${GIT_COMMIT}"
else
	if [[ ! -z $BRANCH ]]; then
		GIT_BRANCH="$BRANCH"
	else
		GIT_BRANCH=unknown
	fi
	GIT_VER="$GIT_BRANCH"
fi

if ! command -v redis-server > /dev/null; then
	echo "Cannot find redis-server. Aborting."
	exit 1
fi 

pack_ramp
[[ $DEPS == 1 ]] && pack_deps

if [[ ! -z $INTO ]]; then
	mkdir -p $INTO
	cd $INTO
	mkdir -p release branch
	
	for f in $RELEASE_ARTIFACTS; do
		[[ -f $BINDIR/$f ]] && cp $BINDIR/$f release/
	done
	
	for f in $DEV_ARTIFACTS; do
		[[ -f $BINDIR/$f ]] && cp $BINDIR/$f branch/
	done
fi

exit 0
