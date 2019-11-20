CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
PROJECT(RedisAI)

# CMake modules should be included in ${PROJECT_SOURCE_DIR}/opt/cmake/modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/opt/cmake/modules)

# Set a default build type if none was specified
set(default_build_type "Release")

IF(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# For adding specific Release flags
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add -fno-omit-frame-pointer to avoid seeing incomplete stack traces
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -ggdb -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb -fno-omit-frame-pointer")

option(BUILD_TF "Build the TensorFlow backend" ON)
option(BUILD_TFLITE "Build the TensorFlow Lite backend" OFF) # Temporarily disabled
option(BUILD_ORT "Build the ONNXRuntime backend" ON)
option(BUILD_TORCH "Build the PyTorch backend" ON)

#----------------------------------------------------------------------------------------------

FUNCTION(ADD_LDFLAGS _TARGET NEW_FLAGS)
    GET_TARGET_PROPERTY(LD_FLAGS ${_TARGET} LINK_FLAGS)
    IF(LD_FLAGS)
        SET(NEW_FLAGS "${LD_FLAGS} ${NEW_FLAGS}")
    ENDIF()
    SET_TARGET_PROPERTIES(${_TARGET} PROPERTIES LINK_FLAGS ${NEW_FLAGS})
ENDFUNCTION()

MACRO(install_symlink filepath sympath)
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
ENDMACRO(install_symlink)

#----------------------------------------------------------------------------------------------

IF (NOT DEVICE)
    SET(DEVICE cpu)
ENDIF()

IF (NOT DEPS_PATH)
    IF (APPLE)
        SET(DEPS_PATH deps/macosx-x64-${DEVICE})
    ELSE()
        SET(DEPS_PATH deps/linux-x64-${DEVICE})
    ENDIF()
ENDIF()

IF (NOT INSTALL_PATH)
    SET(INSTALL_PATH ${CMAKE_SOURCE_DIR}/install-${DEVICE})
ENDIF()

GET_FILENAME_COMPONENT(depsAbs
        "${DEPS_PATH}" REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})

GET_FILENAME_COMPONENT(installAbs
        "${INSTALL_PATH}" REALPATH BASE_DIR ${CMAKE_SOURCE_DIR})

#----------------------------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${depsAbs}/dlpack/include)
IF(BUILD_TF)
    INCLUDE_DIRECTORIES(${depsAbs}/libtensorflow/include)
ENDIF()
IF(BUILD_TFLITE)
    INCLUDE_DIRECTORIES(${depsAbs}/libtensorflow-lite/include)
ENDIF()
IF(BUILD_TORCH)
    INCLUDE_DIRECTORIES(${depsAbs}/libtorch/include)
ENDIF()
IF(BUILD_ORT)
    INCLUDE_DIRECTORIES(${depsAbs}/onnxruntime/include)
ENDIF()

SET(CMAKE_C_STANDARD 11)
ADD_DEFINITIONS(-DREDISMODULE_EXPERIMENTAL_API)

# SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)

#----------------------------------------------------------------------------------------------

IF(BUILD_TF)
    FIND_LIBRARY(TF_LIBRARIES NAMES tensorflow
            PATHS ${depsAbs}/libtensorflow/lib)
    IF (NOT TF_LIBRARIES)
        MESSAGE(WARNING "Could not find TensorFlow in ${depsAbs}/libtensorflow/lib. Trying find_package method")
        FIND_PACKAGE(TensorFlow REQUIRED)
        SET(TF_LIBRARIES ${TensorFlow_LIBRARY})
        IF (NOT TF_LIBRARIES)
            MESSAGE(FATAL_ERROR "Could not find TensorFlow")
        ELSE()
            MESSAGE(STATUS "Found TensorFlow Libraries: \"${TF_LIBRARIES}\")")
        ENDIF()
    ENDIF()
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TFLITE)
    FIND_LIBRARY(TFLITE_LIBRARIES NAMES tensorflow-lite
            PATHS ${depsAbs}/libtensorflow-lite/lib)
    MESSAGE(STATUS "Found TensorFlow Lite Libraries: \"${TFLITE_LIBRARIES}\")")
    IF (NOT TFLITE_LIBRARIES)
        MESSAGE(FATAL_ERROR "Could not find TensorFlow Lite")
    ENDIF()
    IF (${DEVICE} STREQUAL "gpu")
        ADD_DEFINITIONS(-DRAI_TFLITE_USE_CUDA)
    ENDIF()
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_ORT)
    FIND_LIBRARY(ORT_LIBRARIES NAMES onnxruntime
            PATHS ${depsAbs}/onnxruntime/lib)
    MESSAGE(STATUS "Found ONNXRuntime Libraries: \"${ORT_LIBRARIES}\")")
    IF (NOT ORT_LIBRARIES)
        MESSAGE(FATAL_ERROR "Could not find ONNXRuntime")
    ENDIF()
    IF (${DEVICE} STREQUAL "gpu")
        ADD_DEFINITIONS(-DRAI_ONNXRUNTIME_USE_CUDA)
    ENDIF()
ENDIF()

#----------------------------------------------------------------------------------------------

IF (APPLE)
    FIND_LIBRARY(MKL_LIBRARIES NAMES mklml
            PATHS ${depsAbs}/mkl/lib)
    IF (NOT MKL_LIBRARIES)
        MESSAGE(FATAL_ERROR "Could not find MKL for Mac")
    ENDIF()
    SET(platDeps "${MKL_LIBRARIES}")
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TFLITE)
    # Find TensorFlow Lite stuff and build our wrapper
    INCLUDE_DIRECTORIES(util/libtflite_c)

    ADD_SUBDIRECTORY(src/libtflite_c)
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TORCH)
    # Find Torch stuff and build our wrapper
    SET (Torch_DIR ${depsAbs}/libtorch/share/cmake/Torch)
    FIND_PACKAGE(Torch REQUIRED)

    INCLUDE_DIRECTORIES(util/libtorch_c)

    ADD_SUBDIRECTORY(src/libtorch_c)
ENDIF()

#----------------------------------------------------------------------------------------------

ADD_SUBDIRECTORY(src)
ADD_LIBRARY(redisai SHARED $<TARGET_OBJECTS:redisai_obj>)

TARGET_LINK_LIBRARIES(redisai ${CMAKE_DL_LIBS})

SET_TARGET_PROPERTIES(redisai PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(redisai PROPERTIES SUFFIX ".so")

IF (APPLE)
    SET_TARGET_PROPERTIES(redisai PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()

SET(CMAKE_INSTALL_PREFIX ${installAbs})

INSTALL(TARGETS redisai LIBRARY DESTINATION .)

IF (APPLE)
    SET_TARGET_PROPERTIES(redisai PROPERTIES INSTALL_RPATH "@loader_path/lib")
ELSE ()
    ADD_LDFLAGS(redisai "-Wl,--enable-new-dtags")
    SET_TARGET_PROPERTIES(redisai PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
ENDIF()

IF (APPLE)
    SET(LIB_PATTERN "*.dylib")
ELSE()
    SET(LIB_PATTERN "*.so*")
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TF)
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
    INSTALL(DIRECTORY ${depsAbs}/libtensorflow/lib DESTINATION ${installAbs}/backends/redisai_tensorflow
            FILES_MATCHING PATTERN ${LIB_PATTERN})
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TFLITE)
    ADD_LIBRARY(redisai_tflite SHARED $<TARGET_OBJECTS:redisai_tflite_obj>)
    TARGET_LINK_LIBRARIES(redisai_tflite tflite_c ${TFLITE_LIBRARIES})
    SET_TARGET_PROPERTIES(redisai_tflite PROPERTIES PREFIX "")
    SET_TARGET_PROPERTIES(redisai_tflite PROPERTIES SUFFIX ".so")
    IF (APPLE)
        SET_TARGET_PROPERTIES(redisai_tflite PROPERTIES INSTALL_RPATH "@loader_path/lib")
        SET_TARGET_PROPERTIES(redisai_tflite PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    ELSE()
        ADD_LDFLAGS(redisai_tflite "-Wl,--enable-new-dtags")
        SET_TARGET_PROPERTIES(redisai_tflite PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
    ENDIF()
    INSTALL(TARGETS redisai_tflite LIBRARY DESTINATION backends/redisai_tflite)
    INSTALL(DIRECTORY ${depsAbs}/libtensorflow-lite/lib DESTINATION ${installAbs}/backends/redisai_tflite
            FILES_MATCHING PATTERN ${LIB_PATTERN})
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_TORCH)
    ADD_LIBRARY(redisai_torch SHARED $<TARGET_OBJECTS:redisai_torch_obj>)
    TARGET_LINK_LIBRARIES(redisai_torch torch_c ${TORCH_LIBRARIES})
    SET_TARGET_PROPERTIES(redisai_torch PROPERTIES PREFIX "")
    SET_TARGET_PROPERTIES(redisai_torch PROPERTIES SUFFIX ".so")
    IF (APPLE)
        SET_TARGET_PROPERTIES(redisai_torch PROPERTIES INSTALL_RPATH "@loader_path/lib")
    ELSE()
        ADD_LDFLAGS(redisai_torch "-Wl,--enable-new-dtags")
        SET_TARGET_PROPERTIES(redisai_torch PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
    ENDIF()
    INSTALL(TARGETS redisai_torch LIBRARY DESTINATION backends/redisai_torch)
    INSTALL(DIRECTORY ${depsAbs}/libtorch/lib DESTINATION ${installAbs}/backends/redisai_torch
            FILES_MATCHING PATTERN ${LIB_PATTERN})
ENDIF()

#----------------------------------------------------------------------------------------------

IF(BUILD_ORT)
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
    INSTALL(DIRECTORY ${depsAbs}/onnxruntime/lib DESTINATION ${installAbs}/backends/redisai_onnxruntime
            FILES_MATCHING PATTERN ${LIB_PATTERN})
ENDIF()

#----------------------------------------------------------------------------------------------

IF (APPLE)
    INSTALL(DIRECTORY ${depsAbs}/mkl/lib DESTINATION ${installAbs}/backends/redisai_torch
            FILES_MATCHING PATTERN ${LIB_PATTERN})
ENDIF()

#----------------------------------------------------------------------------------------------

IF (NOT ${installAbs} STREQUAL ${CMAKE_SOURCE_DIR}/install-${DEVICE})
    INSTALL_SYMLINK(${installAbs} ${CMAKE_SOURCE_DIR}/install-${DEVICE})
ENDIF()
