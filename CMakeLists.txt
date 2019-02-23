CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
PROJECT(RedisAI)

IF (NOT DEPS_PATH)
    # dependencies are required!
    MESSAGE(FATAL_ERROR "DEPS PATH Missing!")
ENDIF()

GET_FILENAME_COMPONENT(depsAbs
    "${DEPS_PATH}" REALPATH
    BASE_DIR "${CMAKE_BINARY_DIR}")

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(depsInstallPath ${depsAbs}/install/)
INCLUDE_DIRECTORIES(${depsAbs}/dlpack/include)

LIST(APPEND CMAKE_PREFIX_PATH "${depsAbs}/install")
LIST(APPEND CMAKE_PREFIX_PATH "${depsAbs}/libtorch")

FIND_LIBRARY(TF_LIBRARIES
    NAMES tensorflow libtensorflow.so
    PATHS ${depsInstallPath}/lib
    NO_DEFAULT_PATH)

IF (NOT TF_LIBRARIES)
    MESSAGE(FATAL_ERROR "Could not find tensorflow")
ENDIF()

INCLUDE_DIRECTORIES(${depsAbs}/libtensorflow/include)

FILE(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/rminclude)
FILE(COPY ${depsAbs}/redis/src/redismodule.h
    DESTINATION ${PROJECT_BINARY_DIR}/rminclude)
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/rminclude)

# Find Torch stuff and build our wrapper
FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(util/libtorch_c)
ADD_DEFINITIONS(-DREDISMODULE_EXPERIMENTAL_API)

ADD_SUBDIRECTORY(util/libtorch_c)
ADD_SUBDIRECTORY(src)
ADD_LIBRARY(redisai SHARED $<TARGET_OBJECTS:redisai_obj>)
TARGET_LINK_LIBRARIES(redisai torch_c ${TF_LIBRARIES})

IF (APPLE)
    SET_TARGET_PROPERTIES(redisai
        PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()