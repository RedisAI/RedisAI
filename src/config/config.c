#include "config.h"
#include <string.h>
#include "redismodule.h"
#include "backends/backends.h"

// Default configs:
char *BackendsPath;                       //  Path to backends dir. Default value is set when
                                          //  parsing load_time configs.
long long BackendsIntraOpParallelism = 0; //  number of threads used within an
                                          //  individual op for parallelism.
long long BackendsInterOpParallelism = 0; //  number of threads used for parallelism
                                          //  between independent operations.
long long ModelChunkSize = 535822336;     //  size of chunks used to break up model payloads.
                                          //  default is 511 * 1024 * 1024
long long ThreadPoolSizePerQueue = 1;     //  Number of working threads for device.

long long ModelExecutionTimeout = 5000; //  The maximum time in milliseconds
                                        //  before killing onnx run session.
long long BackendMemoryLimit = 0;       //  The maximum amount of memory in MB
                                        //  that backend is allowed to consume.

static int _Config_LoadTimeParamParse(RedisModuleCtx *ctx, const char *key, const char *val,
                                      RedisModuleString *rsval) {
    int ret = REDISMODULE_OK;
    if (strcasecmp((key), "TF") == 0) {
        ret = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, (val));
    } else if (strcasecmp((key), "TFLITE") == 0) {
        ret = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, (val));
    } else if (strcasecmp((key), "TORCH") == 0) {
        ret = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, (val));
    } else if (strcasecmp((key), "ONNX") == 0) {
        ret = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, (val));
    }
    // enable configuring the main thread to create a fixed number of worker
    // threads up front per device. by default we'll use 1
    else if (strcasecmp((key), "THREADS_PER_QUEUE") == 0) {
        ret = Config_SetQueueThreadsNum(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, (val));
        }
    } else if (strcasecmp((key), "INTRA_OP_PARALLELISM") == 0) {
        ret = Config_SetIntraOperationParallelism(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_INTRA_OP_PARALLELISM, val);
        }
    } else if (strcasecmp((key), "INTER_OP_PARALLELISM") == 0) {
        ret = Config_SetInterOperationParallelism(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_INTER_OP_PARALLELISM, val);
        }
    } else if (strcasecmp((key), "MODEL_CHUNK_SIZE") == 0) {
        ret = Config_SetModelChunkSize(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_MODEL_CHUNK_SIZE, val);
        }
    } else if (strcasecmp((key), "MODEL_EXECUTION_TIMEOUT") == 0) {
        ret = Config_SetModelExecutionTimeout(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_MODEL_EXECUTION_TIMEOUT, val);
        }
    } else if (strcasecmp((key), "BACKEND_MEMORY_LIMIT") == 0) {
        ret = Config_SetBackendMemoryLimit(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_BACKEND_MEMORY_LIMIT, val);
        }
    } else if (strcasecmp((key), "BACKENDSPATH") == 0) {
        // already taken care of
    } else {
        ret = REDISMODULE_ERR;
    }
    return ret;
}

long long Config_GetBackendsInterOpParallelism() { return BackendsInterOpParallelism; }

long long Config_GetBackendsIntraOpParallelism() { return BackendsIntraOpParallelism; }

long long Config_GetModelChunkSize() { return ModelChunkSize; }

long long Config_GetNumThreadsPerQueue() { return ThreadPoolSizePerQueue; }

long long Config_GetModelExecutionTimeout() { return ModelExecutionTimeout; }

long long Config_GetBackendMemoryLimit() { return BackendMemoryLimit; }

char *Config_GetBackendsPath() { return BackendsPath; }

int Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc < 3)
        return RedisModule_WrongArity(ctx);

    const char *backend = RedisModule_StringPtrLen(argv[1], NULL);
    const char *path = RedisModule_StringPtrLen(argv[2], NULL);

    int result = REDISMODULE_ERR;
    if (!strcasecmp(backend, "TF")) {
        result = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, path);
    } else if (!strcasecmp(backend, "TFLITE")) {
        result = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, path);
    } else if (!strcasecmp(backend, "TORCH")) {
        result = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, path);
    } else if (!strcasecmp(backend, "ONNX")) {
        result = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, path);
    } else {
        return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
    }
    if (result == REDISMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "OK");
    }
    return RedisModule_ReplyWithError(ctx, "ERR error loading backend");
}

void Config_SetBackendsPath(const char *path) {
    if (BackendsPath != NULL) {
        RedisModule_Free(BackendsPath);
    }
    BackendsPath = RedisModule_Strdup(path);
}

int Config_SetQueueThreadsNum(RedisModuleString *num_threads_string) {
    long long val;
    int result = RedisModule_StringToLongLong(num_threads_string, &val);
    if (result != REDISMODULE_OK || val <= 0) {
        return REDISMODULE_ERR;
    }
    ThreadPoolSizePerQueue = val;
    return REDISMODULE_OK;
}

int Config_SetInterOperationParallelism(RedisModuleString *num_threads_string) {
    long long val;
    int result = RedisModule_StringToLongLong(num_threads_string, &val);
    if (result != REDISMODULE_OK || val <= 0) {
        return REDISMODULE_ERR;
    }
    BackendsInterOpParallelism = val;
    return REDISMODULE_OK;
}

int Config_SetIntraOperationParallelism(RedisModuleString *num_threads_string) {
    long long val;
    int result = RedisModule_StringToLongLong(num_threads_string, &val);
    if (result != REDISMODULE_OK || val <= 0) {
        return REDISMODULE_ERR;
    }
    BackendsIntraOpParallelism = val;
    return REDISMODULE_OK;
}

int Config_SetModelChunkSize(RedisModuleString *chunk_size_string) {
    long long val;
    int result = RedisModule_StringToLongLong(chunk_size_string, &val);
    if (result != REDISMODULE_OK || val <= 0) {
        return REDISMODULE_ERR;
    }
    ModelChunkSize = val;
    return REDISMODULE_OK;
}

int Config_SetModelExecutionTimeout(RedisModuleString *timeout) {
    long long val;
    int result = RedisModule_StringToLongLong(timeout, &val);
    // Timeout should not be lower than the time passing between two consecutive
    // runs of Redis cron callback, that is no more than (1/CONFIG_MIN_HZ)
    if (result != REDISMODULE_OK || val < 1000) {
        return REDISMODULE_ERR;
    }
    ModelExecutionTimeout = val;
    return REDISMODULE_OK;
}

int Config_SetBackendMemoryLimit(RedisModuleString *memory_limit) {
    long long val;
    int result = RedisModule_StringToLongLong(memory_limit, &val);
    if (result != REDISMODULE_OK || val <= 0) {
        return REDISMODULE_ERR;
    }
    BackendMemoryLimit = val;
    return REDISMODULE_OK;
}

int Config_SetLoadTimeParams(RedisModuleCtx *ctx, RedisModuleString *const *argv, int argc) {
    if (argc > 0 && argc % 2 != 0) {
        RedisModule_Log(ctx, "warning",
                        "Even number of arguments provided to module. Please "
                        "provide arguments as KEY VAL pairs");
        return REDISMODULE_ERR;
    }
    // need BACKENDSPATH set up before loading specific backends.
    BackendsPath = RAI_GetBackendsDefaultPath();
    for (int i = 0; i < argc / 2; i++) {
        const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
        const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);
        if (strcasecmp(key, "BACKENDSPATH") == 0) {
            Config_SetBackendsPath(val);
        }
    }

    for (int i = 0; i < argc / 2; i++) {
        const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
        const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);
        int ret = _Config_LoadTimeParamParse(ctx, key, val, argv[2 * i + 1]);

        if (ret == REDISMODULE_ERR) {
            char *buffer = RedisModule_Alloc(
                (4 + strlen(REDISAI_ERRORMSG_PROCESSING_ARG) + strlen(key) + strlen(val)) *
                sizeof(*buffer));
            sprintf(buffer, "%s: %s %s", REDISAI_ERRORMSG_PROCESSING_ARG, key, val);
            RedisModule_Log(ctx, "warning", "%s", buffer);
            RedisModule_Free(buffer);
            return ret;
        }
    }
    return REDISMODULE_OK;
}
