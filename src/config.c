#include "config.h"

#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "backends.h"
#include "background_workers.h"
#include "err.h"
#include "redismodule.h"
#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

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
 *
 * @param queueThreadsString string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_QueueThreads(RedisModuleString *queueThreadsString) {
  int result =
      RedisModule_StringToLongLong(queueThreadsString, &perqueueThreadPoolSize);
  // make sure the number of threads is a positive integer
  // if not set the value to the default
  if (result == REDISMODULE_OK && perqueueThreadPoolSize < 1) {
    perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    result = REDISMODULE_ERR;
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
int RAI_configParamParse(const RedisModuleCtx *ctx, const char *key,
                         const char *val) {
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
    ret = RedisAI_Config_QueueThreads(
        RedisModule_CreateString(NULL, val, strlen(val)));
    if (ret == REDISMODULE_OK) {
      char *buffer = RedisModule_Alloc(
          (3 + strlen(REDISAI_INFOMSG_THREADS_PER_QUEUE) + strlen((val))) *
          sizeof(*buffer));
      sprintf(buffer, "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, (val));
      RedisModule_Log(ctx, "verbose", buffer);
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
int RAI_loadTimeConfig(const RedisModuleCtx *ctx,
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
    int ret = RAI_configParamParse(ctx, key, val);

    if (ret == REDISMODULE_ERR) {
      char *buffer =
          RedisModule_Alloc((4 + strlen(REDISAI_ERRORMSG_PROCESSING_ARG) +
                             strlen(key) + strlen(val)) *
                            sizeof(*buffer));
      sprintf(buffer, "%s: %s %s", REDISAI_ERRORMSG_PROCESSING_ARG, key, val);
      RedisModule_Log(ctx, "warning", buffer);
      RedisModule_Free(buffer);
    }
  }
}
