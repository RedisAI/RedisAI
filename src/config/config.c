#include "config.h"

#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "redismodule.h"
#include "rmutil/alloc.h"
#include "backends/util.h"
#include "backends/backends.h"
#include "util/dict.h"
#include "util/queue.h"
#include "util/arr.h"
#include "execution/background_workers.h"

long long backends_intra_op_parallelism; //  number of threads used within an
                                         //  individual op for parallelism.
long long backends_inter_op_parallelism; //  number of threads used for parallelism
                                         //  between independent operations.
long long model_chunk_size;              //  size of chunks used to break up model payloads.

long long onnx_max_runtime; //  The maximum time in milliseconds
                            //  before killing onnx run session.

static int _RAIConfig_LoadTimeParamParse(RedisModuleCtx *ctx, const char *key, const char *val,
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
        ret = RedisAI_Config_QueueThreads(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, (val));
        }
    } else if (strcasecmp((key), "INTRA_OP_PARALLELISM") == 0) {
        ret = RedisAI_Config_IntraOperationParallelism(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %lld", REDISAI_INFOMSG_INTRA_OP_PARALLELISM,
                            getBackendsIntraOpParallelism());
        }
    } else if (strcasecmp((key), "INTER_OP_PARALLELISM") == 0) {
        ret = RedisAI_Config_InterOperationParallelism(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %lld", REDISAI_INFOMSG_INTER_OP_PARALLELISM,
                            getBackendsInterOpParallelism());
        }
    } else if (strcasecmp((key), "MODEL_CHUNK_SIZE") == 0) {
        ret = RedisAI_Config_ModelChunkSize(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %lld", REDISAI_INFOMSG_MODEL_CHUNK_SIZE,
                            getModelChunkSize());
        }
    } else if (strcasecmp((key), "ONNX_TIMEOUT") == 0) {
        ret = RedisAI_Config_OnnxTimeout(rsval);
        if (ret == REDISMODULE_OK) {
            RedisModule_Log(ctx, "notice", "%s: %lld", REDISAI_INFOMSG_ONNX_TIMEOUT,
                            GetOnnxTimeout());
        }
    } else if (strcasecmp((key), "BACKENDSPATH") == 0) {
        // already taken care of
    } else {
        ret = REDISMODULE_ERR;
    }
    return ret;
}

long long getBackendsInterOpParallelism() { return backends_inter_op_parallelism; }

int setBackendsInterOpParallelism(long long num_threads) {
    if (num_threads <= 0) {
        return REDISMODULE_ERR;
    }
    backends_inter_op_parallelism = num_threads;
    return REDISMODULE_OK;
}

long long getBackendsIntraOpParallelism() { return backends_intra_op_parallelism; }

int setBackendsIntraOpParallelism(long long num_threads) {
    if (num_threads <= 0) {
        return REDISMODULE_ERR;
    }
    backends_intra_op_parallelism = num_threads;
    return REDISMODULE_OK;
}

long long getModelChunkSize() { return model_chunk_size; }

int setModelChunkSize(long long size) {
    if (size <= 0) {
        return REDISMODULE_ERR;
    }
    model_chunk_size = size;
    return REDISMODULE_OK;
}

long long GetNumThreadsPerQueue() { return ThreadPoolSizePerQueue; }

int SetNumThreadsPerQueue(long long num_threads) {
    if (num_threads <= 0) {
        return REDISMODULE_ERR;
    }
    ThreadPoolSizePerQueue = num_threads;
    return REDISMODULE_OK;
}

long long GetOnnxTimeout() { return onnx_max_runtime; }

int SetOnnxTimeout(long long timeout) {
    // Timeout should not be lower than the time passing between two consecutive
    // runs of Redis cron callback, that is no more than (1/CONFIG_MIN_HZ)
    if (timeout < 1000) {
        return REDISMODULE_ERR;
    }
    onnx_max_runtime = timeout;
    return REDISMODULE_OK;
}

int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
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

void RedisAI_Config_BackendsPath(const char *path) {
    if (RAI_BackendsPath != NULL) {
        RedisModule_Free(RAI_BackendsPath);
    }
    RAI_BackendsPath = RedisModule_Strdup(path);
}

int RedisAI_Config_QueueThreads(RedisModuleString *num_threads_string) {
    long long temp;
    int result = RedisModule_StringToLongLong(num_threads_string, &temp);
    if (result != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    return SetNumThreadsPerQueue(temp);
}

int RedisAI_Config_InterOperationParallelism(RedisModuleString *num_threads_string) {
    long long temp;
    int result = RedisModule_StringToLongLong(num_threads_string, &temp);
    if (result != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    return setBackendsInterOpParallelism(temp);
}

int RedisAI_Config_IntraOperationParallelism(RedisModuleString *num_threads_string) {
    long long temp;
    int result = RedisModule_StringToLongLong(num_threads_string, &temp);
    if (result != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    return setBackendsIntraOpParallelism(temp);
}

int RedisAI_Config_ModelChunkSize(RedisModuleString *chunk_size_string) {
    long long temp;
    int result = RedisModule_StringToLongLong(chunk_size_string, &temp);
    if (result != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    return setModelChunkSize(temp);
}

int RedisAI_Config_OnnxTimeout(RedisModuleString *onnx_timeout) {
    long long temp;
    int result = RedisModule_StringToLongLong(onnx_timeout, &temp);
    if (result != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    return SetOnnxTimeout(temp);
}

int RAI_loadTimeConfig(RedisModuleCtx *ctx, RedisModuleString *const *argv, int argc) {
    if (argc > 0 && argc % 2 != 0) {
        RedisModule_Log(ctx, "warning",
                        "Even number of arguments provided to module. Please "
                        "provide arguments as KEY VAL pairs");
        return REDISMODULE_ERR;
    }

    // need BACKENDSPATH set up before loading specific backends
    for (int i = 0; i < argc / 2; i++) {
        const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
        const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);
        if (strcasecmp(key, "BACKENDSPATH") == 0) {
            RedisAI_Config_BackendsPath(val);
        }
    }

    for (int i = 0; i < argc / 2; i++) {
        const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
        const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);
        int ret = _RAIConfig_LoadTimeParamParse(ctx, key, val, argv[2 * i + 1]);

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
