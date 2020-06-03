#!/bin/bash

[[ $IGNERR == 1 ]] || set -e
[[ $VERBOSE == 1 ]] && set -x

[[ -z $DEVICE ]] && { echo DEVICE undefined; exit 1; }
[[ -z $BINDIR ]] && { echo BINDIR undefined; exit 1; }
[[ -z $INSTALL_DIR ]] && { echo INSTALL_DIR undefined; exit 1; }

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
	local FQ_PACKAGE
	if [[ -z $BRANCH ]]; then
		FQ_PACKAGE=$STEM.$VERSION
	else
		FQ_PACKAGE=$STEM.$BRANCH
	fi

	# this is only to extract {semantic_version} into VERSION
	RAMPOUT=$(mktemp /tmp/ramp.XXXXXX)
	$RAMP_PROG pack -m $ROOT/ramp.yml --packname-file $RAMPOUT -o $BINDIR/$PRODUCT.{os}-{architecture}.{semantic_version}.zip $INSTALL_DIR/$PRODUCT.so
	local rampfile=`realpath $(tail -1 $RAMPOUT)`
	rm -f $rampfile $RAMPOUT
	echo `basename $rampfile` | sed -e "s/[^.]*\.[^.]*\.\(.*\)\.zip/\1/" > $BINDIR/VERSION
	export SEMVER=$(cat $BINDIR/VERSION)

	$RAMP_PROG pack -m $ROOT/ramp.yml -o $BINDIR/$FQ_PACKAGE.zip \
		-c "BACKENDSPATH $REDIS_ENT_LIB_PATH/$PRODUCT-$DEVICE-$SEMVER/backends" $INSTALL_DIR/$PRODUCT.so > /dev/null 2>&1

	cd "$BINDIR"
	if [[ -z $BRANCH ]]; then
		ln -sf $FQ_PACKAGE.zip $STEM.latest.zip
	fi

	echo "Done."
}

pack_deps() {
	echo "Building dependencies file ..."

	cd $ROOT

	SEMVER=$(cat $BINDIR/VERSION)

	local STEM=$PRODUCT-$DEVICE-dependencies.$OS-$OSNICK-$ARCH
	local FQ_PACKAGE
	if [[ -z $BRANCH ]]; then
		FQ_PACKAGE=$STEM.$VERSION
	else
		FQ_PACKAGE=$STEM.$BRANCH
	fi
	
	cd $INSTALL_DIR
	local BACKENDS_DIR=$PRODUCT-$DEVICE-$SEMVER
	if [[ ! -h backends ]]; then
		[[ ! -d backends ]] && { echo "install-$DEVICE/backend directory not found." ; exit 1; }
		rm -rf $BACKENDS_DIR
		mkdir $BACKENDS_DIR
	
		mv backends $BACKENDS_DIR
		ln -s $BACKENDS_DIR/backends backends
	fi
	find $BACKENDS_DIR -name "*.so*" | xargs tar pczf $BINDIR/$FQ_PACKAGE.tgz
	
	cd "$BINDIR"
	if [[ -z $BRANCH ]]; then
		ln -sf $FQ_PACKAGE.tgz $STEM.latest.tgz
	fi
	
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
		RAMP=1            build RAMP file
		DEPS=1            build dependencies file
		RAMP_PROG         path to RAMP program

	END
	exit 0
fi

PRODUCT=redisai
PRODUCT_LIB=$PRODUCT.so

if [[ -z $BRANCH ]]; then
	tag=`git describe --abbrev=0 2> /dev/null | sed 's/^v\(.*\)/\1/'`
	if [[ $? != 0 || -z $tag ]]; then
		BRANCH=`git rev-parse --abbrev-ref HEAD`
		VERSION=
	else
		VERSION=$tag
	fi
else
	VERSION=
fi

# GIT_VER=""
# if [[ -d $ROOT/.git ]]; then
# 	if [[ ! -z $BRANCH ]]; then
# 		GIT_BRANCH="$BRANCH"
# 	else
# 		GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# 	fi
# 	GIT_COMMIT=$(git describe --always --abbrev=7 --dirty="+")
# 	# GIT_VER="${GIT_BRANCH}-${GIT_COMMIT}"
# 	GIT_VER="${GIT_BRANCH}"
# else
# 	if [[ ! -z $BRANCH ]]; then
# 		GIT_BRANCH="$BRANCH"
# 	else
# 		GIT_BRANCH=unknown
# 	fi
# 	GIT_VER="$GIT_BRANCH"
# fi

if ! command -v redis-server > /dev/null; then
	echo "Cannot find redis-server. Aborting."
	exit 1
fi 

pack_ramp
[[ $DEPS == 1 ]] && pack_deps

exit 0
