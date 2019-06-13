ADD_LIBRARY(redisai_obj OBJECT
    backends/onnxruntime.c
    backends/tensorflow.c
    backends/torch.c
    redisai.c
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

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/rmutil)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/util)
