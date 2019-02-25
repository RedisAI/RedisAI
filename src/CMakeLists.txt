ADD_LIBRARY(redisai_obj OBJECT
    backends/tensorflow.c
    backends/torch.c
    redisai.c
    model.c
    err.c
    script.c
    tensor.c
    utils/alloc.c)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})