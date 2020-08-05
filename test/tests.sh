#!/bin/bash

[[ $VERBOSE == 1 ]] && set -x
[[ $IGNERR == 1 ]] || set -e

error() {
	echo "There are errors:"
	gawk 'NR>L-4 && NR<L+4 { printf "%-5d%4s%s\n",NR,(NR==L?">>> ":""),$0 }' L=$1 $0
	exit 1
}

[[ -z $_Dbg_DEBUGGER_LEVEL ]] && trap 'error $LINENO' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. $HERE/../opt/readies/shibumi/functions

export ROOT=$(realpath $HERE/..)

#----------------------------------------------------------------------------------------------

help() {
	cat <<-END
		Run Python tests.
	
		[ARGVARS...] tests.sh [--help|help] [<module-so-path>]
		
		Argument variables:
		VERBOSE=1     Print commands
		IGNERR=1      Do not abort on error
		
		DEVICE=cpu|gpu  Device for testing
		GEN=0|1         General tests
		AOF=0|1         Tests with --test-aof
		SLAVES=0|1      Tests with --test-slaves
		
		TEST=test        Run specific test (e.g. test.py:test_name)
		VALGRIND|VGD=1   Run with Valgrind
		CALLGRIND|CGD=1  Run with Callgrind

	END
}

#----------------------------------------------------------------------------------------------

install_git_lfs() {
	[[ $NO_LFS == 1 ]] && return
	[[ $(git lfs env > /dev/null 2>&1 ; echo $?) != 0 ]] && git lfs install
	git lfs pull
}

#----------------------------------------------------------------------------------------------

check_redis_server() {
	if ! command -v redis-server > /dev/null; then
		echo "Cannot find redis-server. Aborting."
		exit 1
	fi
}

#----------------------------------------------------------------------------------------------

valgrind_config() {
	export VG_OPTIONS="
		-q \
		--leak-check=full \
		--show-reachable=no \
		--show-possibly-lost=no"

	VALGRIND_SUPRESSIONS=$ROOT/opt/redis_valgrind.sup

	RLTEST_ARGS+="\
		--no-output-catch \
		--use-valgrind \
		--vg-no-fail-on-errors \
		--vg-verbose \
		--vg-suppressions $VALGRIND_SUPRESSIONS"
}

#----------------------------------------------------------------------------------------------

run_tests() {
	local title="$1"
	[[ ! -z $title ]] && { $ROOT/opt/readies/bin/sep0; printf "Tests with $title:\n\n"; }
	$OP python3 -m RLTest --clear-logs --module $MODULE $RLTEST_ARGS
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help ]] && { help; exit 0; }

DEVICE=${DEVICE:-cpu}

GEN=${GEN:-1}
SLAVES=${SLAVES:-0}
AOF=${AOF:-0}

GDB=${GDB:-0}

OP=""
[[ $NOP == 1 ]] && OP="echo"

MODULE=${MODULE:-$1}
[[ -z $MODULE || ! -f $MODULE ]] && { echo "Module not found. Aborting."; exit 1; }

RLTEST_ARGS=""

[[ $VALGRIND == 1 || $VGD == 1 ]] && valgrind_config

if [[ ! -z $TEST ]]; then
	RLTEST_ARGS+=" --test $TEST"
	export PYDEBUG=${PYDEBUG:-1}
fi

[[ $VERBOSE == 1 ]] && RLTEST_ARGS+=" -v"

#----------------------------------------------------------------------------------------------

cd $ROOT/test

install_git_lfs
check_redis_server

[[ $GEN == 1 ]]    && run_tests
[[ $CLUSTER == 1 ]] && RLTEST_ARGS+=" --env oss-cluster --shards-count 1" run_tests "--env oss-cluster"
[[ $SLAVES == 1 ]] && RLTEST_ARGS+=" --use-slaves" run_tests "--use-slaves"
[[ $AOF == 1 ]]    && RLTEST_ARGS+=" --use-aof" run_tests "--use-aof"
