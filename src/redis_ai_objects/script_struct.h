#pragma once

#include "config/config.h"
#include "tensor_struct.h"
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
    void *infokey;
    const char** entryPoints;
    AI_dict *entryPointsDict; // A <String, TorchScriptFunctionArgumentType*> dict to map between
                           // function name, and its schema.
} RAI_Script;
