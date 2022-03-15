#pragma once

#include "config/config.h"
#include "tensor_struct.h"
#include "redis_ai_objects/stats.h"
#include "util/dict.h"

typedef enum {
    UNKOWN,
    TENSOR,
    INT,
    FLOAT,
    STRING,
    TENSOR_LIST,
    INT_LIST,
    FLOAT_LIST,
    STRING_LIST
} TorchScriptFunctionArgumentType;

typedef struct RAI_Script {
    void *script;
    char *scriptdef;
    // TODO: scripts do not have placement in PyTorch
    // Placement depends on the inputs, as do outputs
    // We keep it here at the moment, until we have a
    // CUDA allocator for dlpack
    char *devicestr;
    RedisModuleString *tag;
    long long refCount;
    RAI_RunStats *info;
    char **entryPoints;
} RAI_Script;
