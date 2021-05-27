if (CMAKE_BUILD_TYPE STREQUAL Debug)
	SET(DEBUG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../opt/readies/cetara/diag/gdb.c")
endif()

file (GLOB_RECURSE SERIALIZATION_SRC 
        redis_ai_objects/tensor.c
        redis_ai_objects/model.c
        redis_ai_objects/script.c
        backends/backends.c
        stats.c
        config.c
        serialization/*.c)

file (GLOB BACKEND_COMMON_SRC
        backends/util.c
        redis_ai_objects/err.c
        util/dict.c
        util/dictionaries.c
        redis_ai_objects/tensor.c
        util/string_utils.c
        execution/utils.c
        serialization/ai_datatypes.c)

ADD_LIBRARY(redisai_obj OBJECT
        util/dict.c
        util/dictionaries.c
        util/queue.c
        util/string_utils.c
        redisai.c
        execution/command_parser.c
        execution/parsing/deprecated.c
        execution/parsing/dag_parser.c
        execution/parsing/model_commands_parser.c
        execution/parsing/script_commands_parser.c
        execution/parsing/parse_utils.c
        execution/run_info.c
        execution/background_workers.c
        execution/utils.c
        config/config.c
        execution/DAG/dag.c
        execution/DAG/dag_builder.c
        execution/DAG/dag_execute.c
        execution/DAG/dag_op.c
        execution/execution_contexts/modelRun_ctx.c
        execution/execution_contexts/scriptRun_ctx.c
        backends/backends.c
        backends/util.c
        redis_ai_objects/model.c
        redis_ai_objects/err.c
        redis_ai_objects/script.c
        redis_ai_objects/stats.c
        redis_ai_objects/tensor.c
        rmutil/alloc.c
        rmutil/sds.c
        rmutil/args.c
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
