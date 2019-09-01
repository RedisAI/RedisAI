#!/bin/bash

[[ -z $DEVICE ]] && { echo DEVICE undefined; exit 1; }
[[ -z $BINDIR ]] && { echo BINDIR undefined; exit 1; }
[[ -z $INSTALL_DIR ]] && { echo INSTALL_DIR undefined; exit 1; }
[[ ! -z $INTO ]] && INTO=$(realpath $INTO)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $HERE/readies/shibumi/functions
ROOT=$(realpath $HERE/..)

RAMP_PROG=ramp
REDIS_ENT_LIB_PATH=/opt/redislabs/lib

BINDIR=$(realpath $BINDIR)
INSTALL_DIR=$(realpath $INSTALL_DIR)

pack_ramp() {
	echo "Building RAMP file ..."
	cd $ROOT
	RAMPOUT=$(mktemp /tmp/ramp.XXXXXX)
	$RAMP_PROG pack -m $ROOT/ramp.yml -o $BINDIR/$PRODUCT.{os}-{architecture}.{semantic_version}.zip $INSTALL_DIR/$PRODUCT.so 2> /dev/null | grep '.zip' > $RAMPOUT
	realpath $(tail -1 $RAMPOUT) > $BINDIR/PACKAGE
	cat $BINDIR/PACKAGE | sed -e "s/[^.]*\.[^.]*\.\(.*\)\.zip/\1/" > $BINDIR/VERSION
	cat $BINDIR/PACKAGE | sed -e "s/[^.]*\.\([^.]*\)\..*\.zip/\1/" > $BINDIR/OSARCH
	PACKAGE=$(cat $BINDIR/PACKAGE)
	VERSION=$(cat $BINDIR/VERSION)
	OSARCH=$(cat $BINDIR/OSARCH)
	$RAMP_PROG pack -m $ROOT/ramp.yml -o "$BINDIR/$PRODUCT.{os}-{architecture}.{semantic_version}.zip" \
		-c "BACKENDSPATH $REDIS_ENT_LIB_PATH/$PRODUCT-$DEVICE-$VERSION/backends" $INSTALL_DIR/$PRODUCT.so 2> /dev/null | grep '.zip' > $RAMPOUT
	rm -f $RAMPOUT
	export PACK_FNAME="$(basename $PACKAGE)"

	cd "$BINDIR"
	RAMP_STEM=$PRODUCT.$OSARCH
	ln -sf $PACK_FNAME $RAMP_STEM.latest.zip
	[[ ! -z $BRANCH ]] && ln -sf $PACK_FNAME $RAMP_STEM.${BRANCH}.zip
	ln -sf $PACK_FNAME $RAMP_STEM.${GIT_VER}.zip

	export RELEASE_ARTIFACTS="$RELEASE_ARTIFACTS $PACK_FNAME $RAMP_STEM.latest.zip"
	export DEV_ARTIFACTS="$DEV_ARTIFACTS $RAMP_STEM.${BRANCH}.zip $RAMP_STEM.${GIT_VER}.zip"
	
	echo "Done."
}

pack_deps() {
	echo "Building dependencies file ..."

	cd $ROOT
	PACK_FNAME=$(basename `cat $BINDIR/PACKAGE`)
	ARCHOSVER=$(echo "$PACK_FNAME" | sed -e "s/^[^.]*\.\([^.]*\..*\)\.zip/\1/")
	VERSION=$(cat $BINDIR/VERSION)
	cd $INSTALL_DIR
	if [[ ! -h backends ]]; then
		[[ ! -d backends ]] && { echo "install-$DEVICE/backend directory not found." ; exit 1; }
		rm -rf $PRODUCT-$DEVICE-$VERSION
		mkdir $PRODUCT-$DEVICE-$VERSION
	
		mv backends $PRODUCT-$DEVICE-$VERSION
		ln -s $PRODUCT-$DEVICE-$VERSION/backends backends
	fi
	find $PRODUCT-$DEVICE-$VERSION -name "*.so*" | xargs tar pczf $BINDIR/$PRODUCT-$DEVICE-dependencies.$ARCHOSVER.tgz
	export DEPS_FNAME="$PRODUCT-$DEVICE-dependencies.$ARCHOSVER.tgz"
	
	cd "$BINDIR"
	DEPS_STEM=$PRODUCT-$DEVICE-dependencies.$OSARCH
	ln -sf $DEPS_FNAME $DEPS_STEM.latest.tgz
	[[ ! -z $BRANCH ]] && ln -sf $DEPS_FNAME $DEPS_STEM.${BRANCH}.tgz
	ln -sf $DEPS_FNAME $DEPS_STEM.${GIT_VER}.tgz
	
	export RELEASE_ARTIFACTS="$RELEASE_ARTIFACTS $DEPS_FNAME $DEPS_STEM.latest.tgz"
	export DEV_ARTIFACTS="$DEV_ARTIFACTS $DEPS_STEM.${BRANCH}.tgz $DEPS_STEM.${GIT_VER}.tgz"

	echo "Done."
 }

set -e
[[ $VERBOSE == 1 ]] && set -x

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
	GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
	GIT_COMMIT=$(git describe --always --abbrev=7 --dirty="+")
	GIT_VER="${GIT_BRANCH}-${GIT_COMMIT}"
fi

OSX=""
if [[ $($HERE/readies/bin/platform --os) == macosx ]]; then
	# macOS: ramp is installed here
	OSX=1
	export PATH=$PATH:$HOME/Library/Python/2.7/bin
fi

if ! command -v redis-server > /dev/null; then
	echo "Cannot find redis-server. Aborting."
	exit 1
fi 

pack_ramp
[[ $DEPS == 1 ]] && pack_deps

if [[ ! -z $INTO ]]; then
	mkdir -p $INTO/release $INTO/branch
	cd $INTO/release
	for f in $RELEASE_ARTIFACTS; do
		[[ -f $BINDIR/$f ]] && ln -sf $(realpath --relative-to . $BINDIR/$f) $(basename $f)
	done
	
	cd $INTO/branch
	for f in $DEV_ARTIFACTS; do
		[[ -f $BINDIR/$f ]] && ln -sf $(realpath --relative-to . $BINDIR/$f) $(basename $f)
	done
fi

exit 0
