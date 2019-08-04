ADD_LIBRARY(redisai_obj OBJECT
    redisai.c
    backends.c
    model.c
    err.c
    script.c
    tensor.c
    util/alloc.c
    rmutil/sds.c
    rmutil/args.c
    rmutil/heap.c
    rmutil/priority_queue.c
    rmutil/vector.c)

ADD_LIBRARY(redisai_tensorflow_obj OBJECT
    backends/tensorflow.c
    err.c
    tensor.c)

ADD_LIBRARY(redisai_torch_obj OBJECT
    backends/torch.c
    err.c
    tensor.c)

ADD_LIBRARY(redisai_onnxruntime_obj OBJECT
    backends/onnxruntime.c
    err.c
    tensor.c)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/rmutil)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/util)
