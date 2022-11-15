/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/**
 * backends.h
 *
 * Contains the structure and method signatures required to register a new
 * backend to be loaded by the module.
 *
 */

#pragma once

#include "config/config.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/tensor.h"
#include "redis_ai_objects/model.h"
#include "redis_ai_objects/script.h"
#include "execution/execution_contexts/execution_ctx.h"

/*
 * To register a new backend to be loaded by the module, the backend needs to
 * implement the following:
 *
 * * ** model_create_with_nodes **:  A callback function pointer that creates a
 * model given the RAI_ModelOpts and input and output nodes.
 *
 * * ** model_create **:  A callback function pointer that creates a model given
 * the RAI_ModelOpts.
 *
 * * ** model_run **:  A callback function pointer that runs a model given the
 * RAI_Model pointer and an array of RAI_ExecutionCtx pointers.
 *
 * * ** model_serialize **:  A callback function pointer that serializes a model
 * given the RAI_Model pointer.
 *
 * * ** script_create **:  A callback function pointer that creates a script.
 *
 * * ** script_free **:  A callback function pointer that frees a script given
 * the RAI_Script pointer.
 *
 * * ** script_run **:  A callback function pointer that runs a model given the
 * RAI_Script pointer and .
 */
typedef struct RAI_LoadedBackend {
    // ** model_create_with_nodes **:  A callback function pointer that creates a
    // model given the RAI_ModelOpts and input and output nodes
    RAI_Model *(*model_create_with_nodes)(RAI_Backend, const char *, RAI_ModelOpts, size_t,
                                          const char **, size_t, const char **, const char *,
                                          size_t, RAI_Error *);

    // ** model_create **:  A callback function pointer that creates a model given
    // the RAI_ModelOpts
    RAI_Model *(*model_create)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                               RAI_Error *);

    // ** model_free **:  A callback function pointer that frees a model given the
    // RAI_Model pointer
    void (*model_free)(RAI_Model *, RAI_Error *);

    // ** model_run **:  A callback function pointer that runs a model given the
    // RAI_Model pointer and an array of RAI_ExecutionCtx pointers
    int (*model_run)(RAI_Model *, RAI_ExecutionCtx **, RAI_Error *);

    // ** model_serialize **:  A callback function pointer that serializes a model
    // given the RAI_Model pointer
    int (*model_serialize)(RAI_Model *, char **, size_t *, RAI_Error *);

    // ** script_create **:  A callback function pointer that creates a script
    RAI_Script *(*script_create)(const char *, const char *, const char **, size_t, RAI_Error *);

    // ** script_free **:  A callback function pointer that frees a script given
    // the RAI_Script pointer
    void (*script_free)(RAI_Script *, RAI_Error *);

    // ** script_run **:  A callback function pointer that runs a model given the
    // RAI_ScriptRunCtx pointer
    int (*script_run)(RAI_Script *, const char *function, RAI_ExecutionCtx *, RAI_Error *);

    // Returns the backend version.
    const char *(*get_version)(void);

    // Returns the backend's memory usage for INFO report
    unsigned long long (*get_memory_info)(void);

    // Returns the number of times that Redis accessed backend allocator.
    unsigned long long (*get_memory_access_num)(void);

    // A callback for to use whenever a new device is introduced.
    int (*add_new_device_cb)(const char *);

    // Kill run session callback (for stopping long runs).
    void (*stop_long_running_sessions_cb)(RedisModuleCtx *, RedisModuleEvent, uint64_t, void *);

    // Get the number of maximum run sessions that can run.
    size_t (*get_max_run_sessions)(void);
} RAI_LoadedBackend;

typedef struct RAI_LoadedBackends {
    RAI_LoadedBackend tf;
    RAI_LoadedBackend tflite;
    RAI_LoadedBackend torch;
    RAI_LoadedBackend onnx;
} RAI_LoadedBackends;

RAI_LoadedBackends RAI_backends;

int RAI_LoadBackend(RedisModuleCtx *ctx, int backend, const char *path);

int RAI_LoadDefaultBackend(RedisModuleCtx *ctx, int backend);

/**
 * @brief Returns the backend name as string.
 */
const char *RAI_GetBackendName(RAI_Backend backend);

/**
 * @brief Set the default backends path (<module_path>/backends) in backends_path place holder.
 */
void RAI_SetBackendsDefaultPath(char **backends_path);
