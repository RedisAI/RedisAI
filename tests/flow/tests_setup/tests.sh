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
export ROOT=$(cd $HERE/../../..; pwd)
. $ROOT/opt/readies/shibumi/functions

cd $HERE

#----------------------------------------------------------------------------------------------

help() {
	cat <<-END
		Run Python tests.
	
		[ARGVARS...] tests.sh [--help|help] [<module-so-path>]
		
		Argument variables:
		VERBOSE=1     Print commands
		IGNERR=1      Do not abort on error

		MODULE=path     Path to redisai.so
		TESTMOD=path    Path to LLAPI module
		
		DEVICE=cpu|gpu  Device for testing
		GEN=0|1         General tests
		SLAVES=0|1      Tests with --test-slaves
		
		TEST=test        Run specific test (e.g. test.py:test_name)
		LOG=0|1          Write to log
		VALGRIND|VGD=1   Run with Valgrind
		CALLGRIND|CGD=1  Run with Callgrind

	END
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
		--use-valgrind \
		--vg-suppressions $VALGRIND_SUPRESSIONS
		--cluster_node_timeout 60000"
}

valgrind_summary() {
	# Collect name of each flow log that contains leaks
	FILES_WITH_LEAKS=$(grep -l "definitely lost" logs/*.valgrind.log)
	if [[ ! -z $FILES_WITH_LEAKS ]]; then
		echo "Memory leaks introduced in flow tests."
		echo $FILES_WITH_LEAKS
		# Print the full Valgrind output for each leaking file
		echo $FILES_WITH_LEAKS | xargs cat
		exit 1
	else
		echo Valgrind test ok
	fi
}

#----------------------------------------------------------------------------------------------

get_tests_data() {
	local TEST_DATA_PATH=$ROOT/tests/flow/test_data
	if [ ! -d ${TEST_DATA_PATH} ]; then
	  echo "Downloading tests data from s3..."
	  wget -q -x -nH --cut-dirs=2 -i $ROOT/tests/flow/test_data_files.txt -P $ROOT/tests/flow/test_data
	  echo "Done"
	fi
}

#----------------------------------------------------------------------------------------------

run_tests() {
	local title="$1"
	[[ ! -z $title ]] && { $ROOT/opt/readies/bin/sep -0; printf "Tests with $title:\n\n"; }
	cd $ROOT/tests/flow
	$OP python3 -m RLTest --clear-logs --module $MODULE $RLTEST_ARGS
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help ]] && { help; exit 0; }

DEVICE=${DEVICE:-cpu}

GEN=${GEN:-1}
SLAVES=${SLAVES:-0}

GDB=${GDB:-0}

OP=""
[[ $NOP == 1 ]] && OP="echo"

MODULE=${MODULE:-$1}
[[ -z $MODULE || ! -f $MODULE ]] && { echo "Module not found at ${MODULE}. Aborting."; exit 1; }
TESTMOD=${TESTMOD}
echo "Test module path is ${TESTMOD}"

[[ $VALGRIND == 1 || $VGD == 1 ]] && valgrind_config

if [[ ! -z $TEST ]]; then
	RLTEST_ARGS+=" --test $TEST"
	if [[ $LOG != 1 ]]; then
		RLTEST_ARGS+=" -s"
		export PYDEBUG=${PYDEBUG:-1}
	fi
fi

[[ $VERBOSE == 1 ]] && RLTEST_ARGS+=" -v"
[[ $GDB == 1 ]] && RLTEST_ARGS+=" -i --verbose"

#----------------------------------------------------------------------------------------------

cd $ROOT/tests/flow/tests_setup

check_redis_server
./Install_RedisGears.sh
get_tests_data

[[ ! -z $REDIS ]] && RL_TEST_ARGS+=" --env exiting-env --existing-env-addr $REDIS" run_tests "redis-server: $REDIS"
[[ $GEN == 1 ]]    && run_tests
[[ $CLUSTER == 1 ]] && RLTEST_ARGS+=" --env oss-cluster --shards-count 3" run_tests "--env oss-cluster"
[[ $VALGRIND != 1 && $SLAVES == 1 ]] && RLTEST_ARGS+=" --use-slaves" run_tests "--use-slaves"
# [[ $VALGRIND == 1 ]] && valgrind_summary
exit 0
