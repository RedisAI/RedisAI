#pragma once

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

#define RAI_COPY_RUN_OUTPUT
#define RAI_PRINT_BACKEND_ERRORS

#define REDISAI_ERRORMSG_PROCESSING_ARG "ERR error processing argument"

#define REDISAI_INFOMSG_THREADS_PER_QUEUE       "Setting THREADS_PER_QUEUE parameter to"
#define REDISAI_INFOMSG_INTRA_OP_PARALLELISM    "Setting INTRA_OP_PARALLELISM parameter to"
#define REDISAI_INFOMSG_INTER_OP_PARALLELISM    "Setting INTER_OP_PARALLELISM parameter to"
#define REDISAI_INFOMSG_MODEL_CHUNK_SIZE        "Setting MODEL_CHUNK_SIZE parameter to"
#define REDISAI_INFOMSG_MODEL_EXECUTION_TIMEOUT "Setting MODEL_EXECUTION_TIMEOUT parameter to"
#define REDISAI_INFOMSG_BACKEND_MEMORY_LIMIT    "Setting BACKEND_MEMORY_LIMIT parameter to"

/**
 * Get number of threads used for parallelism between independent operations, by
 * backend.
 */
long long Config_GetBackendsInterOpParallelism(void);

/**
 * Get number of threads used within an individual op for parallelism, by
 * backend.
 */
long long Config_GetBackendsIntraOpParallelism(void);

/**
 * @return size of chunks (in bytes) in which models are split for
 * set, get, serialization and replication.
 */
long long Config_GetModelChunkSize(void);

/**
 * @brief Return the number of working threads per device in RedisAI.
 */
long long Config_GetNumThreadsPerQueue(void);

/**
 * @return Number of milliseconds that a model session is allowed to run
 * before killing it. Currently supported only for onnxruntime backend.
 */
long long Config_GetModelExecutionTimeout(void);

/**
 * @return Memory limit in MB for backend. This is the maximum amount of memory
 * that can be consumed by the backend for creating and running sessions.
 * Currently supported only for onnxruntime backend.
 */
long long Config_GetBackendMemoryLimit(void);

/**
 * @return Returns the backends path string.
 */
char *Config_GetBackendsPath(void);

/**
 * Helper method for AI.CONFIG LOADBACKEND <backend_identifier>
 * <location_of_backend_library>
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR otherwise.
 */
int Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

/**
 * Helper method for AI.CONFIG BACKENDSPATH
 * <default_location_of_backend_libraries>
 * @param path string containing backend path
 */
void Config_SetBackendsPath(const char *path);

/**
 * Set number of threads used for parallelism between RedisAI independent
 * blocking commands (AI.DAGEXECUTE, AI.SCRIPTEXECUTE, AI.MODELEXECUTE).
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed
 */
int Config_SetQueueThreadsNum(RedisModuleString *num_threads_string);

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int Config_SetInterOperationParallelism(RedisModuleString *num_threads_string);

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int Config_SetIntraOperationParallelism(RedisModuleString *num_threads_string);

/**
 * Set size of chunks in which model payloads are split for set,
 * get, serialization and replication.
 * @param chunk_size_string string containing chunk size (in bytes)
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed
 */
int Config_SetModelChunkSize(RedisModuleString *chunk_size_string);

/**
 * Set the maximum time in ms that onnx backend allow running a model.
 * @param timeout - string containing the max runtime (in ms)
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int Config_SetModelExecutionTimeout(RedisModuleString *timeout);

/**
 * Set the memory limit in MB for backends allocations.
 * @param memory_limit - maximum memory consumption by backend. If values is zero,
 * there will be no enforcement of any memory limit.
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int Config_SetBackendMemoryLimit(RedisModuleString *memory_limit);

/**
 * Load time configuration parser
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if failed
 */
int Config_SetLoadTimeParams(RedisModuleCtx *ctx, RedisModuleString *const *argv, int argc);
