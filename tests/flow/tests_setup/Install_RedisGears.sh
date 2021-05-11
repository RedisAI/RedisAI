#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT=$(cd $HERE/../../.. && pwd)
READIES=$ROOT/opt/readies
. $READIES/shibumi/defs

if [[ "$1" == "--help" || "$1" == "help" || "$HELP" == "1" ]]; then
	cat <<-END
		Obtain RedisGears module binaries

		Install_RedisGears.sh [--help|help]

		Argument variables:
		GEARS_OSNICK=nick   Get binaries for give osnick
		GEARS_PATH=dir      Get binaries from given Gears repo
		NOP=1               No operation
		HELP=1              Show help

	END
	exit 0
fi

OP=""
[[ "$NOP" == "1" ]] && OP=echo

os="$($READIES/bin/platform --os)"
arch="$($READIES/bin/platform --arch)"

if [[ ! -z "$GEARS_PATH" ]]; then
	platform="$($READIES/bin/platform -t)"
else
	if [[ "$os" != "linux" || "$arch" != "x64" ]]; then
		eprint "Cannot match binary artifacts - build RedisGears and set GEARS_PATH"
		exit 1
	fi

	dist="$($READIES/bin/platform --dist)"
	nick="$($READIES/bin/platform --osnick)"

	if [[ $dist == "ubuntu" ]]; then
		if [[ $nick != "bionic" && $nick != "xenial" && $nick != "trusty" ]]; then
			nick="bionic"
		fi
	elif [[ $dist == debian ]]; then
		nick=bionic
	elif [[ $dist == centos || $dist == redhat || $dist == fedora ]]; then
		nick=centos7
	elif [[ ! -z "$GEARS_OSNICK" ]]; then
		nick=$GEARS_OSNICK
	else
		eprint "Cannot match binary artifacts - build RedisGears and set GEARS_PATH"
		exit 1
	fi
	platform="${os}-${nick}-${arch}"
fi

GEARS_S3_URL="http://redismodules.s3.amazonaws.com/redisgears/snapshots"
GEARS_MOD="redisgears.${platform}.master.zip"
GEARS_DEPS="redisgears-python.${platform}.master.tgz"

FINAL_WORK_DIR="$ROOT/bin/$($READIES/bin/platform -t)/RedisGears"

if [[ -d $FINAL_WORK_DIR && -f $FINAL_WORK_DIR/redisgears.so ]]; then
	echo "RedisGears is in ${FINAL_WORK_DIR}"
	exit 0
fi

$OP mkdir -p $(dirname $FINAL_WORK_DIR)
$OP rm -rf ${FINAL_WORK_DIR}.*
WORK_DIR=$(mktemp -d ${FINAL_WORK_DIR}.XXXXXX)
$OP mkdir -p $WORK_DIR

if [[ -z $GEARS_PATH ]]; then
	F_GEARS_MOD="$WORK_DIR/$GEARS_MOD"
	if [[ ! -f $F_GEARS_MOD ]]; then
		echo "Download RedisGears ..."
		$OP wget -q -P $WORK_DIR $GEARS_S3_URL/$GEARS_MOD
	fi

	F_GEARS_DEPS="$WORK_DIR/$GEARS_DEPS"
	if [[ ! -f $F_GEARS_DEPS ]]; then
		echo "Download RedisGears deps ..."
		$OP wget -q -P $WORK_DIR $GEARS_S3_URL/$GEARS_DEPS
	fi
else
	F_GEARS_MOD="${GEARS_PATH}/artifacts/snapshot/${GEARS_MOD}"
	F_GEARS_DEPS="${GEARS_PATH}/artifacts/snapshot/${GEARS_DEPS}"
	[[ ! -f $F_GEARS_MOD ]] && { eprint "$F_GEARS_MOD is missing"; exit 1; }
	[[ ! -f $F_GEARS_DEPS ]] && { eprint "$F_GEARS_DEPS is missing"; exit 1; }
fi

$OP unzip -q $F_GEARS_MOD -d $WORK_DIR
$OP tar --no-same-owner -C $WORK_DIR -xzf $F_GEARS_DEPS
$OP mv $WORK_DIR $FINAL_WORK_DIR
