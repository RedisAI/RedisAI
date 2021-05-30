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

//#define RAI_COPY_RUN_INPUT
#define RAI_COPY_RUN_OUTPUT
#define RAI_PRINT_BACKEND_ERRORS
#define REDISAI_DEFAULT_THREADS_PER_QUEUE     1
#define REDISAI_DEFAULT_INTRA_OP_PARALLELISM  0
#define REDISAI_DEFAULT_INTER_OP_PARALLELISM  0
#define REDISAI_DEFAULT_MODEL_CHUNK_SIZE      535822336 // (511 * 1024 * 1024)
#define REDISAI_ERRORMSG_PROCESSING_ARG       "ERR error processing argument"
#define REDISAI_ERRORMSG_THREADS_PER_QUEUE    "ERR error setting THREADS_PER_QUEUE to"
#define REDISAI_ERRORMSG_INTRA_OP_PARALLELISM "ERR error setting INTRA_OP_PARALLELISM to"
#define REDISAI_ERRORMSG_INTER_OP_PARALLELISM "ERR error setting INTER_OP_PARALLELISM to"

#define REDISAI_INFOMSG_THREADS_PER_QUEUE    "Setting THREADS_PER_QUEUE parameter to"
#define REDISAI_INFOMSG_INTRA_OP_PARALLELISM "Setting INTRA_OP_PARALLELISM parameter to"
#define REDISAI_INFOMSG_INTER_OP_PARALLELISM "Setting INTER_OP_PARALLELISM parameter to"
#define REDISAI_INFOMSG_MODEL_CHUNK_SIZE     "Setting MODEL_CHUNK_SIZE parameter to"

/**
 * Get number of threads used for parallelism between independent operations, by
 * backend.
 * @return number of threads used for parallelism between independent
 * operations, by backend
 */
long long getBackendsInterOpParallelism();

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsInterOpParallelism(long long num_threads);

/**
 * Get number of threads used within an individual op for parallelism, by
 * backend.
 * @return number of threads used within an individual op for parallelism, by
 * backend.
 */
long long getBackendsIntraOpParallelism();

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsIntraOpParallelism(long long num_threads);

/**
 * @return size of chunks (in bytes) in which models are split for
 * set, get, serialization and replication.
 */
long long getModelChunkSize();

/**
 * Set size of chunks (in bytes) in which models are split for set,
 * get, serialization and replication.
 *
 * @param size
 * @return 0 on success, or 1  if failed
 */
int setModelChunkSize(long long size);

/**
 * Helper method for AI.CONFIG LOADBACKEND <backend_identifier>
 * <location_of_backend_library>
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

/**
 * Helper method for AI.CONFIG BACKENDSPATH
 * <default_location_of_backend_libraries>
 *
 * @param ctx Context in which Redis modules operate
 * @param path string containing backend path
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RedisAI_Config_BackendsPath(RedisModuleCtx *ctx, const char *path);

/**
 * Set number of threads used for parallelism between RedisAI independent
 * blocking commands ( AI.DAGRUN, AI.SCRIPTRUN, AI.MODELRUN ).
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_QueueThreads(RedisModuleString *num_threads_string);

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_InterOperationParallelism(RedisModuleString *num_threads_string);

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads_string string containing thread number
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_IntraOperationParallelism(RedisModuleString *num_threads_string);

/**
 * Set size of chunks in which model payloads are split for set,
 * get, serialization and replication.
 *
 * @param chunk_size_string string containing chunk size (in bytes)
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RedisAI_Config_ModelChunkSize(RedisModuleString *chunk_size_string);

/**
 *
 * @param ctx Context in which Redis modules operate
 * @param key
 * @param val
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if failed
 */
int RAI_configParamParse(RedisModuleCtx *ctx, const char *key, const char *val,
                         RedisModuleString *rsval);

/**
 * Load time configuration parser
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR  if the DAGRUN failed
 */
int RAI_loadTimeConfig(RedisModuleCtx *ctx, RedisModuleString *const *argv, int argc);
