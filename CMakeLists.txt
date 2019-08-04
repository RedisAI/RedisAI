CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
PROJECT(RedisAI)

IF (NOT DEPS_PATH)
    # dependencies are required!
    MESSAGE(FATAL_ERROR "DEPS PATH Missing!")
ENDIF()

FUNCTION(ADD_LDFLAGS _TARGET NEW_FLAGS)
    GET_TARGET_PROPERTY(LD_FLAGS ${_TARGET} LINK_FLAGS)
    IF(LD_FLAGS)
        SET(NEW_FLAGS "${LD_FLAGS} ${NEW_FLAGS}")
    ENDIF()
    SET_TARGET_PROPERTIES(${_TARGET} PROPERTIES LINK_FLAGS ${NEW_FLAGS})
ENDFUNCTION()

GET_FILENAME_COMPONENT(depsAbs
    "${DEPS_PATH}" REALPATH
    BASE_DIR "${CMAKE_BINARY_DIR}")

INCLUDE_DIRECTORIES(${depsAbs}/dlpack/include)
INCLUDE_DIRECTORIES(${depsAbs}/libtensorflow/include)
INCLUDE_DIRECTORIES(${depsAbs}/libtorch/include)
INCLUDE_DIRECTORIES(${depsAbs}/onnxruntime/include)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CMAKE_C_STANDARD 99)

FIND_LIBRARY(TF_LIBRARIES NAMES tensorflow libtensorflow.so
             PATHS ${depsAbs}/libtensorflow/lib)
IF (NOT TF_LIBRARIES)
    MESSAGE(FATAL_ERROR "Could not find tensorflow")
ENDIF()

FIND_LIBRARY(ORT_LIBRARIES NAMES onnxruntime
             PATHS ${depsAbs}/onnxruntime/lib)
IF (NOT ORT_LIBRARIES)
    MESSAGE(FATAL_ERROR "Could not find ONNXRuntime")
ENDIF()

IF (APPLE)
    FIND_LIBRARY(MKL_LIBRARIES NAMES mklml
                 PATHS ${depsAbs}/onnxruntime/lib)
    IF (NOT MKL_LIBRARIES)
        MESSAGE(FATAL_ERROR "Could not find MKL for Mac")
    ENDIF()
    SET(TORCH_LIBRARIES "${MKL_LIBRARIES}")
ENDIF()

# Find Torch stuff and build our wrapper
SET (Torch_DIR ${depsAbs}/libtorch/share/cmake/Torch)
FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(util/libtorch_c)
ADD_DEFINITIONS(-DREDISMODULE_EXPERIMENTAL_API)

ADD_SUBDIRECTORY(util/libtorch_c)
ADD_SUBDIRECTORY(src)
ADD_LIBRARY(redisai SHARED $<TARGET_OBJECTS:redisai_obj>)

TARGET_LINK_LIBRARIES(redisai ${CMAKE_DL_LIBS})

SET_TARGET_PROPERTIES(redisai PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(redisai PROPERTIES SUFFIX ".so")

IF (APPLE)
    SET_TARGET_PROPERTIES(redisai PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

INSTALL(TARGETS redisai LIBRARY DESTINATION .)

ADD_LIBRARY(redisai_tensorflow SHARED $<TARGET_OBJECTS:redisai_tensorflow_obj>)
TARGET_LINK_LIBRARIES(redisai_tensorflow ${TF_LIBRARIES})
SET_TARGET_PROPERTIES(redisai_tensorflow PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(redisai_tensorflow PROPERTIES SUFFIX ".so")
IF (APPLE)
  SET_TARGET_PROPERTIES(redisai_tensorflow PROPERTIES INSTALL_RPATH "@loader_path/lib")
ELSE ()
  ADD_LDFLAGS(redisai_tensorflow "-Wl,--enable-new-dtags")
  SET_TARGET_PROPERTIES(redisai_tensorflow PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
ENDIF()
INSTALL(TARGETS redisai_tensorflow LIBRARY DESTINATION backends/redisai_tensorflow)
INSTALL(DIRECTORY ${depsAbs}/libtensorflow/lib DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/redisai_tensorflow)

ADD_LIBRARY(redisai_torch SHARED $<TARGET_OBJECTS:redisai_torch_obj>)
TARGET_LINK_LIBRARIES(redisai_torch torch_c ${TORCH_LIBRARIES})
SET_TARGET_PROPERTIES(redisai_torch PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(redisai_torch PROPERTIES SUFFIX ".so")
IF (APPLE)
  SET_TARGET_PROPERTIES(redisai_torch PROPERTIES INSTALL_RPATH "@loader_path/lib")
ELSE ()
  ADD_LDFLAGS(redisai_torch "-Wl,--enable-new-dtags")
  SET_TARGET_PROPERTIES(redisai_torch PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
ENDIF()
INSTALL(TARGETS redisai_torch LIBRARY DESTINATION backends/redisai_torch)
INSTALL(DIRECTORY ${depsAbs}/libtorch/lib DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/redisai_torch)

ADD_LIBRARY(redisai_onnxruntime SHARED $<TARGET_OBJECTS:redisai_onnxruntime_obj>)
TARGET_LINK_LIBRARIES(redisai_onnxruntime ${ORT_LIBRARIES})
SET_TARGET_PROPERTIES(redisai_onnxruntime PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(redisai_onnxruntime PROPERTIES SUFFIX ".so")
IF (APPLE)
  SET_TARGET_PROPERTIES(redisai_onnxruntime PROPERTIES INSTALL_RPATH "@loader_path/lib")
ELSE ()
  ADD_LDFLAGS(redisai_onnxruntime "-Wl,--enable-new-dtags")
  SET_TARGET_PROPERTIES(redisai_onnxruntime PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
ENDIF()
INSTALL(TARGETS redisai_onnxruntime LIBRARY DESTINATION backends/redisai_onnxruntime)
INSTALL(DIRECTORY ${depsAbs}/onnxruntime/lib DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/redisai_onnxruntime)
