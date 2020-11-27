#!/bin/bash

# Collect name of each flow log that contains leaks
FILES_WITH_LEAKS=$(grep -l "definitely lost: [1-9][0-9,]* bytes" logs/*.valgrind.log)
if [[ ! -z $FILES_WITH_LEAKS ]]; then
	echo "Memory leaks introduced in flow tests."
	# Print the full Valgrind output for each leaking file
	echo $FILES_WITH_LEAKS | xargs cat
	exit 1
else
	echo Valgrind test ok
fi
