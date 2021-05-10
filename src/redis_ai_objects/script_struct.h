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
    AI_dict *functionData; // A <String, TorchScriptFunctionArgumentType*> dict to map between
                           // function name, and its schema.
} RAI_Script;

typedef struct RAI_ScriptCtxParam {
    RAI_Tensor *tensor;
} RAI_ScriptCtxParam;

typedef struct RAI_ScriptRunCtx {
    size_t ctxtype;
    RAI_Script *script;
    char *fnname;
    RAI_ScriptCtxParam *inputs;
    RAI_ScriptCtxParam *outputs;
    size_t *listSizes;
    int32_t *intInputs;
    float *floatInputs;
    RedisModuleString **stringInputs;
} RAI_ScriptRunCtx;
