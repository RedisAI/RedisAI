#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#include "redismodule.h"

typedef enum { RAI_MODEL, RAI_SCRIPT } RAI_RunType;

typedef enum {
  RAI_BACKEND_TENSORFLOW = 0,
  RAI_BACKEND_TFLITE,
  RAI_BACKEND_TORCH,
  RAI_BACKEND_ONNXRUNTIME,
} RAI_Backend;

// NOTE: entry in queue hash is formed by
// device * MAX_DEVICE_ID + deviceid

typedef enum { RAI_DEVICE_CPU = 0, RAI_DEVICE_GPU = 1 } RAI_Device;

#define RAI_ENC_VER 900

//#define RAI_COPY_RUN_INPUT
#define RAI_COPY_RUN_OUTPUT
#define RAI_PRINT_BACKEND_ERRORS
#define REDISAI_DEFAULT_THREADS_PER_QUEUE 1

/**
 * Helper method for AI.CONFIG LOADBACKEND <backend_identifier> <location_of_backend_library>
 * 
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv,
                               int argc);

/**
 * Helper method for AI.CONFIG BACKENDSPATH <default_location_of_backend_libraries>
 * 
 * @param ctx Context in which Redis modules operate
 * @param path string containing backend path
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_BackendsPath(RedisModuleCtx *ctx, const char *path);

/**
 *
 * @param queueThreadsString string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_QueueThreads(RedisModuleString *queueThreadsString);

/**
 *
 * @param ctx Context in which Redis modules operate
 * @param key
 * @param val
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RAI_configParamParse(const RedisModuleCtx *ctx, const char *key,
                         const char *val);

/**
 * Load time configuration parser
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RAI_loadTimeConfig(const RedisModuleCtx *ctx,
                       RedisModuleString *const *argv, int argc);

#endif /* SRC_CONFIG_H_ */
