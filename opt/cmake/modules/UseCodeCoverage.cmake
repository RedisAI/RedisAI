# - Enable Code Coverage
#
# Variables you may define are:
#  CODECOV_HTMLOUTPUTDIR - the name of the directory where HTML results are placed. Defaults to "coverage_html"
#  CODECOV_XMLOUTPUTFILE - the name of the directory where HTML results are placed. Defaults to "coverage.xml"
#  CODECOV_GCOVR_OPTIONS - additional options given to gcovr commands.
#

if(ENABLE_CODECOVERAGE)

    if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
        message( WARNING "Code coverage results with an optimised (non-Debug) build may be misleading" )
    endif ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )

    if ( NOT DEFINED CODECOV_OUTPUTFILE )
        set( CODECOV_OUTPUTFILE coverage.info )
    endif ( NOT DEFINED CODECOV_OUTPUTFILE )

    if ( NOT DEFINED CODECOV_HTMLOUTPUTDIR )
        set( CODECOV_HTMLOUTPUTDIR coverage_results )
    endif ( NOT DEFINED CODECOV_HTMLOUTPUTDIR )

    if ( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCXX )
        find_program( CODECOV_GCOV gcov )
        find_program( CODECOV_LCOV lcov )
        find_program( CODECOV_GENHTML genhtml )
        add_definitions( -fprofile-arcs -ftest-coverage )
        link_libraries( gcov )
        set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} --coverage )
        add_custom_target( coverage_init ALL ${CODECOV_LCOV} --base-directory .  --directory ${CMAKE_BINARY_DIR}/src --output-file ${CODECOV_OUTPUTFILE} --capture --initial )
        add_custom_target( coverage ${CODECOV_LCOV} --base-directory .  --directory ${CMAKE_BINARY_DIR}/src --output-file ${CODECOV_OUTPUTFILE} --capture COMMAND genhtml -o ${CODECOV_HTMLOUTPUTDIR} ${CODECOV_OUTPUTFILE} )
    endif ( CMAKE_COMPILER_IS_GNUCXX )

endif(ENABLE_CODECOVERAGE)
