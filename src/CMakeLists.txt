if (CMAKE_BUILD_TYPE STREQUAL Debug)
	SET(DEBUG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../opt/readies/cetara/diag/gdb.c")
endif()

file (GLOB_RECURSE SERIALIZATION_SRC 
        tensor/tensor.c
        model/model.c
        script/script.c
        backends.c
        stats.c
        config.c
        serialization/*.c)

file (GLOB BACKEND_COMMON_SRC
        backends/util.c
        err.c
        util/dict.c
        tensor/tensor.c
        util/string_utils.c
        serialization/ai_datatypes.c)

ADD_LIBRARY(redisai_obj OBJECT
        util/dict.c
        util/queue.c
        util/string_utils.c
        redisai.c
        command_parser.c
        run_info.c
        background_workers.c
        config.c
        DAG/dag.c
        DAG/dag_parser.c
        DAG/dag_builder.c
        DAG/dag_execute.c
        model/modelRun_ctx.c
        backends.c
        backends/util.c
        model/model.c
        err.c
        script/script.c
        stats.c
        tensor/tensor.c
        rmutil/alloc.c
        rmutil/sds.c
        rmutil/args.c
        rmutil/heap.c
        rmutil/priority_queue.c
        rmutil/vector.c run_info.c
        redis_ai_types/model_type.c
        redis_ai_types/tensor_type.c
        redis_ai_types/script_type.c
        ${SERIALIZATION_SRC}
		${DEBUG_SRC})

IF(BUILD_TF)
    ADD_LIBRARY(redisai_tensorflow_obj OBJECT
            backends/tensorflow.c
            ${BACKEND_COMMON_SRC}
)
ENDIF()

IF(BUILD_TFLITE)
    ADD_LIBRARY(redisai_tflite_obj OBJECT
            backends/tflite.c
            ${BACKEND_COMMON_SRC}
)
ENDIF()

IF(BUILD_TORCH)
    ADD_LIBRARY(redisai_torch_obj OBJECT
            backends/torch.c
            ${BACKEND_COMMON_SRC}
)
ENDIF()

IF(BUILD_ORT)
    ADD_LIBRARY(redisai_onnxruntime_obj OBJECT
            backends/onnxruntime.c
            ${BACKEND_COMMON_SRC}
)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/rmutil)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/util)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/model)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/script)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/tensor)
