if (CMAKE_BUILD_TYPE STREQUAL Debug)
	SET(DEBUG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../opt/readies/cetara/diag/gdb.c")
endif()

ADD_LIBRARY(redisai_obj OBJECT
        util/dict.c
        util/queue.c
        redisai.c
        run_info.c
        background_workers.c
        config.c
        dag.c
        dag_parser.c
        modelRun_ctx.c
        dag_llapi.c
        backends.c
        backends/util.c
        model.c
        err.c
        script.c
        stats.c
        tensor.c
        rmutil/alloc.c
        rmutil/sds.c
        rmutil/args.c
        rmutil/heap.c
        rmutil/priority_queue.c
        rmutil/vector.c run_info.c
		${DEBUG_SRC})

IF(BUILD_TF)
    ADD_LIBRARY(redisai_tensorflow_obj OBJECT
            backends/tensorflow.c
            backends/util.c
            err.c
            util/dict.c
            tensor.c)
ENDIF()

IF(BUILD_TFLITE)
    ADD_LIBRARY(redisai_tflite_obj OBJECT
            backends/tflite.c
            backends/util.c
            err.c
            util/dict.c
            tensor.c)
ENDIF()

IF(BUILD_TORCH)
    ADD_LIBRARY(redisai_torch_obj OBJECT
            backends/torch.c
            backends/util.c
            err.c
            util/dict.c
            tensor.c)
ENDIF()

IF(BUILD_ORT)
    ADD_LIBRARY(redisai_onnxruntime_obj OBJECT
            backends/onnxruntime.c
            backends/util.c
            err.c
            util/dict.c
            tensor.c)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/rmutil)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/util)
