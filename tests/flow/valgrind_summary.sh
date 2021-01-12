#!/bin/bash
FILES_WITH_LEAKS=$(grep -l "definitely lost" logs/*.valgrind.log)
	if [[ ! -z $FILES_WITH_LEAKS ]]; then
		echo "Memory leaks introduced in flow tests."
		echo $FILES_WITH_LEAKS
		# Print the full Valgrind output for each leaking file
		echo $FILES_WITH_LEAKS | xargs cat
	else
		echo Valgrind test ok
	fi