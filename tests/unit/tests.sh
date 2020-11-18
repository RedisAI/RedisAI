#!/bin/bash

[[ $VERBOSE == 1 ]] && set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export ROOT=$(realpath $HERE/../..)

#----------------------------------------------------------------------------------------------

help() {
	cat <<-END
		Run Unit tests.
	
		[ARGVARS...] tests.sh [--help|help]
		
		Argument variables:

	END
}


#----------------------------------------------------------------------------------------------

run_tests() {
	make -C $BINDIR test
}

#----------------------------------------------------------------------------------------------

[[ $1 == --help || $1 == help ]] && { help; exit 0; }


BINDIR=${BINDIR:-""}

[[ ! -d $BINDIR ]] && { echo "BINDIR not provided. Aborting."; exit 1; }


#----------------------------------------------------------------------------------------------

cd $ROOT/tests/unit

run_tests
