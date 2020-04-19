ADD_LIBRARY(redisai_obj OBJECT
        util/dict.c
        util/queue.c
        redisai.c
        run_info.c
        background_workers.c
        model_script_run_session.c
        config.c
        dag.c
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
        rmutil/vector.c run_info.c)

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
