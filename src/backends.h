/**
 * backends.h
 *
 * Contains the structure and method signatures required to register a new
 * backend to be loaded by the module.
 *
 */

#ifndef SRC_BACKENDS_H_
#define SRC_BACKENDS_H_

#include "config.h"
#include "err.h"
#include "model_struct.h"
#include "script_struct.h"
#include "tensor.h"

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
 * RAI_ModelRunCtx pointer.
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
 * RAI_ScriptRunCtx pointer.
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
    // RAI_ModelRunCtx pointer
    int (*model_run)(RAI_ModelRunCtx **, RAI_Error *);

    // ** model_serialize **:  A callback function pointer that serializes a model
    // given the RAI_Model pointer
    int (*model_serialize)(RAI_Model *, char **, size_t *, RAI_Error *);

    // ** script_create **:  A callback function pointer that creates a script
    RAI_Script *(*script_create)(const char *, const char *, RAI_Error *);

    // ** script_free **:  A callback function pointer that frees a script given
    // the RAI_Script pointer
    void (*script_free)(RAI_Script *, RAI_Error *);

    // ** script_run **:  A callback function pointer that runs a model given the
    // RAI_ScriptRunCtx pointer
    int (*script_run)(RAI_ScriptRunCtx *, RAI_Error *);

    // Returns the backend version.
    const char *(*get_version)(void);

} RAI_LoadedBackend;

typedef struct RAI_LoadedBackends {
    RAI_LoadedBackend tf;
    RAI_LoadedBackend tflite;
    RAI_LoadedBackend torch;
    RAI_LoadedBackend onnx;
} RAI_LoadedBackends;

RAI_LoadedBackends RAI_backends;
char *RAI_BackendsPath;

int RAI_LoadBackend(RedisModuleCtx *ctx, int backend, const char *path);
int RAI_LoadDefaultBackend(RedisModuleCtx *ctx, int backend);

const char *RAI_BackendName(int backend);

#endif
