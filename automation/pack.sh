#!/bin/bash

[[ -z $DEVICE ]] && { echo DEVICE undefined; exit 1; }
[[ -z $BINDIR ]] && { echo BINDIR undefined; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $HERE/../automation/readies/shibumi/functions
ROOT=$(realpath $HERE/..)

RAMP_PROG=ramp
REDIS_ENT_LIB_PATH=/opt/redislabs/lib

BINDIR=$(realpath $BINDIR)

pack_ramp() {
	echo "Building RAMP file ..."
	cd $ROOT
	RAMPOUT=$(mktemp /tmp/ramp.XXXXXX)
	echo BINDIR=$BINDIR
	$RAMP_PROG pack -m $ROOT/ramp.yml -o $BINDIR/$PRODUCT.{os}-{architecture}.{semantic_version}.zip $INSTALL_DIR/redisai.so 2> /dev/null | grep '.zip' > $RAMPOUT
	realpath $(tail -1 $RAMPOUT) > $BINDIR/PACKAGE
	cat $BINDIR/PACKAGE | sed -e "s/[^.]*\.[^.]*\.\(.*\)\.zip/\1/" > $BINDIR/VERSION
	VERSION=$(cat $BINDIR/VERSION)
	$RAMP_PROG pack -m $ROOT/ramp.yml -o "build/redisai.{os}-{architecture}.{semantic_version}.zip" \
		-c "BACKENDSPATH $REDIS_ENT_LIB_PATH/redisai-cpu-$VERSION/backends" $INSTALL_DIR/redisai.so 2> /dev/null | grep '.zip' > $RAMPOUT
	rm -f $RAMPOUT
	echo "Done."
}

pack_deps() {
	echo "Building dependencies file ..."

	cd $ROOT
	PACK_FNAME=$(basename `cat $BINDIR/PACKAGE`)
	ARCHOSVER=$(echo "$PACK_FNAME" | sed -e "s/^redisai\.\([^.]*\..*\)\.zip/\1/")
	VERSION=$(cat $BINDIR/VERSION)
	cd install-$DEVICE
	if [[ ! -h backends ]]; then
		[[ ! -d backends ]] && { echo "install-$DEVICE/backend directory not found." ; exit 1; }
		rm -rf redisai-$DEVICE-$VERSION
		mkdir redisai-$DEVICE-$VERSION
	
		mv backends redisai-$DEVICE-$VERSION
		ln -s redisai-$DEVICE-$VERSION/backends backends
	fi
	find redisai-$DEVICE-$VERSION -name "*.so*" | xargs tar pczf redisai-$DEVICE-dependencies.$ARCHOSVER.tgz
	echo "Done."
 }

set -e
[[ $VERBOSE == 1 ]] && set -x

if [[ $1 == --help || $1 == help ]]; then
	cat <<-END
		pack.sh [cpu|gpu] [--help|help]
		
		Argument variables:
		DEVICE=cpu|gpu
		BRANCH=branch   branch names to serve as an exta package tag
		INTO=dir        package destination directory (optinal)
		RAMP=1          build RAMP file
		DEPS=1          build dependencies file
		RAMP_PROG       path to RAMP program

	END
	exit 0
fi

# BINDIR=$(cat $ROOT/BINDIR)
BIN=$ROOT/bin
INSTALL_DIR=$ROOT/install

PRODUCT=redisai
PRODUCT_LIB=$PRODUCT.so

GIT_VER=""
if [[ -d $ROOT/.git ]]; then
	GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
	GIT_COMMIT=$(git describe --always --abbrev=7 --dirty="+")
	GIT_VER="${GIT_BRANCH}-${GIT_COMMIT}"
fi

# OSX=""
# if [[ $(./deps/readies/bin/platform --os) == macosx ]]; then
# 	OSX=1
# 	export PATH=$PATH:$HOME/Library/Python/2.7/bin
# fi

if ! command -v redis-server > /dev/null; then
	echo "Cannot find redis-server. Aborting."
	exit 1
fi 

pack_ramp
pack_deps
echo "Done."
exit 0


cd "$BINDIR"
ln -s $PACK_FNAME $PRODUCT.latest.zip
ln -s $DEPS_FNAME $PRODUCT-dependencies.latest.tgz

if [[ ! -z $BRANCH ]]; then
	ln -s $PACK_FNAME $PRODUCT.${BRANCH}.zip
	ln -s $DEPS_FNAME $PRODUCT-dependencies.${BRANCH}.tgz
fi
ln -s $PACK_FNAME $PRODUCT.${GIT_VER}.zip
ln -s $DEPS_FNAME $PRODUCT-dependencies.${GIT_VER}.tgz

RELEASE_ARTIFACTS=\
	$PACK_FNAME $DEPS_FNAME \
	$PRODUCT.latest.zip $PRODUCT-dependencies.latest.tgz

DEV_ARTIFACTS=\
	$PRODUCT.${BRANCH}.zip $PRODUCT-dependencies.${BRANCH}.tgz \
	$PRODUCT.${GIT_VER}.zip $PRODUCT-dependencies.${GIT_VER}.tgz

if [[ ! -z $INTO ]]; then
	INTO=$(realpath $INTO)
	mkdir -p $INTO/release $INTO/branch
	cd $INTO/release
	foreach f in ($RELEASE_ARTIFACTS)
		ln -s $f
	end
	
	cd $INTO/branch
	foreach f in ($DEV_ARTIFACTS)
		ln -s $f
	end
fi

echo "Done."
