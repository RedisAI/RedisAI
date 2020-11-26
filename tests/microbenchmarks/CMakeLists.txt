

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/opt/googlebenchmark" "opt/googlebenchmark")
include_directories({PROJECT_SOURCE_DIR}/opt/googlebenchmark/include)

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})

    target_include_directories(${TESTNAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)

    target_link_libraries(${TESTNAME} benchmark::benchmark redisai redisai_torch redisai_tensorflow)

    set_target_properties(${TESTNAME} PROPERTIES FOLDER benchmarks)
endmacro()

package_add_test(microbenchmark_tensor microbenchmark_tensor.cpp)
