#!/bin/bash

[[ $VERBOSE == 1 ]] && set -x
[[ $IGNERR == 1 ]] || set -e

error() {
	echo "There are errors."
	exit 1
}

# trap error ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export ROOT=$(cd $HERE/../..; pwd)
. $ROOT/opt/readies/shibumi/functions

cd $HERE

#----------------------------------------------------------------------------------------------

help() {
	cat <<-END
		Run Valgrind/Callgrind on RedisAI.
	
		[ARGVARS...] valgrind.sh [--help|help] [<module-so-path>]
		
		Argument variables:
		VERBOSE=1         Print commands
		IGNERR=1          Do not abort on error
		
		SUPRESSIONS       Suppressions file
		CALLGRIND|CGD=1   Run with Callgrind

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
	VALGRIND_SUPRESSIONS=$ROOT/opt/redis_valgrind.sup

	if [[ $CALLGRIND == 1 ]]; then
		
		VALGRIND_OPTIONS+="\
			--tool=callgrind \
			--dump-instr=yes \
			--simulate-cache=no \
			--collect-jumps=no \
			--collect-atstart=yes \
			--instr-atstart=yes \
			--callgrind-out-file=$MODULE.call"
	else
		VALGRIND_OPTIONS+="\
			-q \
			--suppressions=$SUPRESSIONS \
			--leak-check=full \
			--show-reachable=no \
			--show-possibly-lost=no \
			--show-leak-kinds=all"
	fi

	VALGRIND_OPTIONS+=" -v"
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help ]] && { help; exit 0; }

OP=""
[[ $NOP == 1 ]] && OP="echo"

MODULE=${MODULE:-$1}
[[ -z $MODULE || ! -f $MODULE ]] && { echo "Module not found. Aborting."; exit 1; }

VALGRIND_OPTIONS=""
valgrind_config

#----------------------------------------------------------------------------------------------

check_redis_server
$OP valgrind $(echo "$VALGRIND_OPTIONS") redis-server --protected-mode no --save '' --appendonly no --loadmodule $MODULE
