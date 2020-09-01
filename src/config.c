#include "config.h"

#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "backends.h"
#include "backends/util.h"
#include "background_workers.h"
#include "err.h"
#include "redismodule.h"
#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

long long backends_intra_op_parallelism;  //  number of threads used within an
//  individual op for parallelism.
long long
    backends_inter_op_parallelism;  //  number of threads used for parallelism
                                    //  between independent operations.

/**
 *
 * @return number of threads used within an individual op for parallelism.
 */
long long getBackendsInterOpParallelism() {
  return backends_inter_op_parallelism;
}

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsInterOpParallelism(long long num_threads) {
  int result = 1;
  if (num_threads >= 0) {
    backends_inter_op_parallelism = num_threads;
    result = 0;
  }
  return result;
}

/**
 *
 * @return
 */
long long getBackendsIntraOpParallelism() {
  return backends_intra_op_parallelism;
}

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsIntraOpParallelism(long long num_threads) {
  int result = 1;
  if (num_threads >= 0) {
    backends_intra_op_parallelism = num_threads;
    result = 0;
  }
  return result;
}

/**
 * Helper method for AI.CONFIG LOADBACKEND <backend_identifier>
 * <location_of_backend_library>
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv,
                               int argc) {
  if (argc < 3) return RedisModule_WrongArity(ctx);

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

/**
 * Helper method for AI.CONFIG BACKENDSPATH
 * <default_location_of_backend_libraries>
 *
 * @param ctx Context in which Redis modules operate
 * @param path string containing backend path
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_BackendsPath(RedisModuleCtx *ctx, const char *path) {
  if (RAI_BackendsPath != NULL) {
    RedisModule_Free(RAI_BackendsPath);
  }
  RAI_BackendsPath = RedisModule_Strdup(path);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/**
 * Set number of threads used for parallelism between RedisAI independent
 * blocking commands ( AI.DAGRUN, AI.SCRIPTRUN, AI.MODELRUN ).
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_QueueThreads(RedisModuleString *num_threads_string) {
  int result =
      RedisModule_StringToLongLong(num_threads_string, &perqueueThreadPoolSize);
  // make sure the number of threads is a positive integer
  // if not set the value to the default
  if (result == REDISMODULE_OK && perqueueThreadPoolSize < 1) {
    perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    result = REDISMODULE_ERR;
  }
  return result;
}

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_InterOperationParallelism(
    RedisModuleString *num_threads_string) {
  long long temp;
  int result = RedisModule_StringToLongLong(num_threads_string, &temp);
  if (result == REDISMODULE_OK) {
    result = setBackendsInterOpParallelism(temp);
  }
  return result;
}

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_IntraOperationParallelism(
    RedisModuleString *num_threads_string) {
  long long temp;
  int result = RedisModule_StringToLongLong(num_threads_string, &temp);
  if (result == REDISMODULE_OK) {
    result = setBackendsIntraOpParallelism(temp);
  }
  return result;
}

/**
 *
 * @param ctx Context in which Redis modules operate
 * @param key
 * @param val
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RAI_configParamParse(RedisModuleCtx *ctx, const char *key,
                         const char *val, RedisModuleString *rsval) {
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
      char *buffer = RedisModule_Alloc(
          (3 + strlen(REDISAI_INFOMSG_THREADS_PER_QUEUE) + strlen((val))) *
          sizeof(*buffer));
      sprintf(buffer, "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, (val));
      RedisModule_Log(ctx, "notice", "%s", buffer);
      RedisModule_Free(buffer);
    }
  } else if (strcasecmp((key), "INTRA_OP_PARALLELISM") == 0) {
    ret = RedisAI_Config_IntraOperationParallelism(rsval);
    if (ret == REDISMODULE_OK) {
      char *buffer = RedisModule_Alloc(
          (3 + strlen(REDISAI_INFOMSG_INTRA_OP_PARALLELISM) + strlen((val))) *
          sizeof(*buffer));
      sprintf(buffer, "%s: %lld", REDISAI_INFOMSG_INTRA_OP_PARALLELISM,
              getBackendsIntraOpParallelism());
      RedisModule_Log(ctx, "notice", "%s", buffer);
      RedisModule_Free(buffer);
    }
  } else if (strcasecmp((key), "INTER_OP_PARALLELISM") == 0) {
    ret = RedisAI_Config_InterOperationParallelism(rsval);
    if (ret == REDISMODULE_OK) {
      char *buffer = RedisModule_Alloc(
          (3 + strlen(REDISAI_INFOMSG_INTER_OP_PARALLELISM) + strlen((val))) *
          sizeof(*buffer));
      sprintf(buffer, "%s: %lld", REDISAI_INFOMSG_INTER_OP_PARALLELISM,
              getBackendsInterOpParallelism());
      RedisModule_Log(ctx, "notice", "%s", buffer);
      RedisModule_Free(buffer);
    }
  } else if (strcasecmp((key), "BACKENDSPATH") == 0) {
    // already taken care of
  } else {
    ret = REDISMODULE_ERR;
  }
  return ret;
}

/**
 * Load time configuration parser
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RAI_loadTimeConfig(RedisModuleCtx *ctx,
                       RedisModuleString *const *argv, int argc) {
  if (argc > 0 && argc % 2 != 0) {
    RedisModule_Log(ctx, "warning",
                    "Even number of arguments provided to module. Please "
                    "provide arguments as KEY VAL pairs");
  }

  // need BACKENDSPATH set up before loading specific backends
  for (int i = 0; i < argc / 2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);

    int ret = REDISMODULE_OK;
    if (strcasecmp(key, "BACKENDSPATH") == 0) {
      ret = RedisAI_Config_BackendsPath(ctx, val);
    }
  }

  for (int i = 0; i < argc / 2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2 * i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2 * i + 1], NULL);
    int ret = RAI_configParamParse(ctx, key, val, argv[2 * i + 1]);

    if (ret == REDISMODULE_ERR) {
      char *buffer =
          RedisModule_Alloc((4 + strlen(REDISAI_ERRORMSG_PROCESSING_ARG) +
                             strlen(key) + strlen(val)) *
                            sizeof(*buffer));
      sprintf(buffer, "%s: %s %s", REDISAI_ERRORMSG_PROCESSING_ARG, key, val);
      RedisModule_Log(ctx, "warning", "%s", buffer);
      RedisModule_Free(buffer);
      return ret;
    }
  }

  return REDISMODULE_OK;
}
