/**
 * backends.c
 *
 * Contains the methods required to register a new backend to be loaded by the
 * module.
 *
 */

#ifdef __linux__
#define _GNU_SOURCE
#endif
#include "backends.h"

#include <dlfcn.h>
#include <libgen.h>
#include <string.h>
#include "redismodule.h"
#include "config/config.h"
#include "execution/background_workers.h"
#include "execution/execution_contexts/modelRun_ctx.h"

static bool _ValidateFuncExists(RedisModuleCtx *ctx, void *func_ptr, const char *func_name,
                                const char *backend_name, const char *path) {
    if (func_ptr == NULL) {
        RedisModule_Log(ctx, "warning",
                        "Backend does not export %s. %s backend"
                        " was not loaded from %s",
                        func_name, backend_name, path);
        return false;
    }
    return true;
}

/**
 * @brief Export a function from RedisAI to a backend. This will set a pointer
 * to a function that has been declared in the backend to use the corresponding
 * function in RedisAI.
 * @param func_name A string that identifies the function to export.
 * @param targetFuncPtr place holder for a function pointer coming from the
 * backend to set the corresponding function from RedisAI into it.
 */
int RAI_ExportFunc(const char *func_name, void **targetFuncPtr) {

    // Retrieve info from RedisAI internals.
    if (strcmp("GetThreadId", func_name) == 0) {
        *targetFuncPtr = BGWorker_GetThreadId;
    } else if (strcmp("GetNumThreadsPerQueue", func_name) == 0) {
        *targetFuncPtr = Config_GetNumThreadsPerQueue;
    } else if (strcmp("GetModelExecutionTimeout", func_name) == 0) {
        *targetFuncPtr = Config_GetModelExecutionTimeout;
    } else if (strcmp("GetThreadsCount", func_name) == 0) {
        *targetFuncPtr = BGWorker_GetThreadsCount;
    } else if (strcmp("GetBackendMemoryLimit", func_name) == 0) {
        *targetFuncPtr = Config_GetBackendMemoryLimit;

        // Export RedisAI low level API functions.
    } else if (strcmp("RedisAI_InitError", func_name) == 0) {
        *targetFuncPtr = RAI_InitError;
    } else if (strcmp("RedisAI_FreeError", func_name) == 0) {
        *targetFuncPtr = RAI_FreeError;
    } else if (strcmp("RedisAI_GetError", func_name) == 0) {
        *targetFuncPtr = RAI_GetError;
    } else if (strcmp("RedisAI_TensorCreateFromDLTensor", func_name) == 0) {
        *targetFuncPtr = RAI_TensorCreateFromDLTensor;
    } else if (strcmp("RedisAI_TensorGetDLTensor", func_name) == 0) {
        *targetFuncPtr = RAI_TensorGetDLTensor;
    } else if (strcmp("RedisAI_TensorGetShallowCopy", func_name) == 0) {
        *targetFuncPtr = RAI_TensorGetShallowCopy;
    } else if (strcmp("RedisAI_TensorFree", func_name) == 0) {
        *targetFuncPtr = RAI_TensorFree;
    } else if (strcmp("RedisAI_GetModelFromKeyspace", func_name) == 0) {
        *targetFuncPtr = RAI_GetModelFromKeyspace;
    } else if (strcmp("RedisAI_ModelRunCtxCreate", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxCreate;
    } else if (strcmp("RedisAI_ModelRunCtxAddInput", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxAddInput;
    } else if (strcmp("RedisAI_ModelRunCtxNumOutputs", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxNumOutputs;
    } else if (strcmp("RedisAI_ModelRunCtxAddOutput", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxAddOutput;
    } else if (strcmp("RedisAI_ModelRunCtxOutputTensor", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxOutputTensor;
    } else if (strcmp("RedisAI_ModelRunCtxFree", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRunCtxFree;
    } else if (strcmp("RedisAI_ModelRun", func_name) == 0) {
        *targetFuncPtr = RAI_ModelRun;

        // Export RedisModule API functions.
    } else {
        return RedisModule_GetApi(func_name, targetFuncPtr);
    }
    return REDISMODULE_OK;
}

void RAI_SetBackendsDefaultPath(char **backends_path) {
    RedisModule_Assert(*backends_path == NULL);
    Dl_info info;
    // Retrieve the info about the module's dynamic library, and extract the .so file dir name.
    RedisModule_Assert(dladdr(RAI_SetBackendsDefaultPath, &info) != 0);
    const char *dyn_lib_dir_name = dirname((char *)info.dli_fname);

    // Populate backends_path global string with the default path.
    size_t backends_default_path_len = strlen(dyn_lib_dir_name) + strlen("/backends");
    *backends_path = RedisModule_Alloc(backends_default_path_len + 1);
    RedisModule_Assert(sprintf(*backends_path, "%s/backends", dyn_lib_dir_name) > 0);
}

const char *RAI_GetBackendName(RAI_Backend backend) {
    switch (backend) {
    case RAI_BACKEND_TENSORFLOW:
        return "TF";
    case RAI_BACKEND_TFLITE:
        return "TFLITE";
    case RAI_BACKEND_TORCH:
        return "TORCH";
    case RAI_BACKEND_ONNXRUNTIME:
        return "ONNX";
    }
    return NULL;
}

int RAI_LoadBackend_TensorFlow(RedisModuleCtx *ctx, const char *path) {
    if (RAI_backends.tf.model_run != NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TF backend: backend already loaded");
        return REDISMODULE_ERR;
    }

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TF backend from %s: %s", path, dlerror());
        return REDISMODULE_ERR;
    }
    RAI_LoadedBackend backend = {0}; // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void *));
    init_backend =
        (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(handle, "RAI_InitBackendTF");
    if (!_ValidateFuncExists(ctx, init_backend, "RAI_InitBackendTF", "TF", path)) {
        goto error;
    }
    // Here we use the input callback to export functions from Redis to the backend,
    // by setting the backend's function pointers to the corresponding functions in Redis.
    init_backend(RedisModule_GetApi);

    backend.model_create_with_nodes =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, size_t, const char **, size_t,
                         const char **, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTF");
    if (!_ValidateFuncExists(ctx, backend.model_create_with_nodes, "RAI_ModelCreateTF", "TF",
                             path)) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTF");
    if (!_ValidateFuncExists(ctx, backend.model_free, "RAI_ModelFreeTF", "TF", path)) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
        unsigned long)dlsym(handle, "RAI_ModelRunTF");
    if (!_ValidateFuncExists(ctx, backend.model_run, "RAI_ModelRunTF", "TF", path)) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        (unsigned long)dlsym(handle, "RAI_ModelSerializeTF"));
    if (!_ValidateFuncExists(ctx, backend.model_serialize, "RAI_ModelSerializeTF", "TF", path)) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTF");
    if (!_ValidateFuncExists(ctx, backend.get_version, "RAI_GetBackendVersionTF", "TF", path)) {
        goto error;
    }

    RAI_backends.tf = backend;
    RedisModule_Log(ctx, "notice", "TF backend loaded from %s", path);
    return REDISMODULE_OK;

error:
    dlclose(handle);
    return REDISMODULE_ERR;
}

int RAI_LoadBackend_TFLite(RedisModuleCtx *ctx, const char *path) {
    if (RAI_backends.tflite.model_run != NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TFLITE backend: backend already loaded");
        return REDISMODULE_ERR;
    }

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);

    if (handle == NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TFLITE backend from %s: %s", path,
                        dlerror());
        return REDISMODULE_ERR;
    }
    RAI_LoadedBackend backend = {0}; // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTFLite");
    if (!_ValidateFuncExists(ctx, init_backend, "RAI_InitBackendTFLite", "TFLite", path)) {
        goto error;
    }
    // Here we use the input callback to export functions from Redis to the backend,
    // by setting the backend's function pointers to the corresponding functions in Redis.
    init_backend(RedisModule_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTFLite");
    if (!_ValidateFuncExists(ctx, backend.model_create, "RAI_ModelCreateTFLite", "TFLite", path)) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTFLite");
    if (!_ValidateFuncExists(ctx, backend.model_free, "RAI_ModelFreeTFLite", "TFLite", path)) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
        unsigned long)dlsym(handle, "RAI_ModelRunTFLite");
    if (!_ValidateFuncExists(ctx, backend.model_run, "RAI_ModelRunTFLite", "TFLite", path)) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTFLite");
    if (!_ValidateFuncExists(ctx, backend.model_serialize, "RAI_ModelSerializeTFLite", "TFLite",
                             path)) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTFLite");
    if (!_ValidateFuncExists(ctx, backend.get_version, "RAI_GetBackendVersionTFLite", "TFLite",
                             path)) {
        goto error;
    }

    RAI_backends.tflite = backend;
    RedisModule_Log(ctx, "notice", "TFLITE backend loaded from %s", path);
    return REDISMODULE_OK;

error:
    dlclose(handle);
    return REDISMODULE_ERR;
}

int RAI_LoadBackend_Torch(RedisModuleCtx *ctx, const char *path) {
    if (RAI_backends.torch.model_run != NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TORCH backend: backend already loaded");
        return REDISMODULE_ERR;
    }

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        RedisModule_Log(ctx, "warning", "Could not load TORCH backend from %s: %s", path,
                        dlerror());
        return REDISMODULE_ERR;
    }

    RAI_LoadedBackend backend = {0}; // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void **));
    init_backend = (int (*)(int (*)(const char *, void **)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTorch");
    if (!_ValidateFuncExists(ctx, init_backend, "RAI_InitBackendTorch", "TORCH", path)) {
        goto error;
    }
    // Here we use the input callback to export functions from Redis and Redis AI to the backend,
    // by setting the backend's function pointers to the corresponding functions in Redis/RedisAI.
    init_backend(RAI_ExportFunc);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTorch");
    if (!_ValidateFuncExists(ctx, backend.model_create, "RAI_ModelCreateTorch", "TORCH", path)) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTorch");
    if (!_ValidateFuncExists(ctx, backend.model_free, "RAI_ModelFreeTorch", "TORCH", path)) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
        unsigned long)dlsym(handle, "RAI_ModelRunTorch");
    if (!_ValidateFuncExists(ctx, backend.model_run, "RAI_ModelRunTorch", "TORCH", path)) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTorch");
    if (!_ValidateFuncExists(ctx, backend.model_serialize, "RAI_ModelSerializeTorch", "TORCH",
                             path)) {
        goto error;
    }

    backend.script_create =
        (RAI_Script * (*)(const char *, const char *, const char **, size_t, RAI_Error *))(
            unsigned long)dlsym(handle, "RAI_ScriptCreateTorch");
    if (!_ValidateFuncExists(ctx, backend.script_create, "RAI_ScriptCreateTorch", "TORCH", path)) {
        goto error;
    }

    backend.script_free =
        (void (*)(RAI_Script *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ScriptFreeTorch");
    if (!_ValidateFuncExists(ctx, backend.script_free, "RAI_ScriptFreeTorch", "TORCH", path)) {
        goto error;
    }

    backend.script_run = (int (*)(RAI_Script *, const char *, RAI_ExecutionCtx *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ScriptRunTorch");
    if (!_ValidateFuncExists(ctx, backend.script_run, "RAI_ScriptRunTorch", "TORCH", path)) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTorch");
    if (!_ValidateFuncExists(ctx, backend.get_version, "RAI_GetBackendVersionTorch", "TORCH",
                             path)) {
        goto error;
    }

    RAI_backends.torch = backend;
    RedisModule_Log(ctx, "notice", "TORCH backend loaded from %s", path);
    return REDISMODULE_OK;

error:
    dlclose(handle);
    return REDISMODULE_ERR;
}

int RAI_LoadBackend_ONNXRuntime(RedisModuleCtx *ctx, const char *path) {
    if (RAI_backends.onnx.model_run != NULL) {
        RedisModule_Log(ctx, "warning", "Could not load ONNX backend: backend already loaded");
        return REDISMODULE_ERR;
    }

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);

    if (handle == NULL) {
        RedisModule_Log(ctx, "warning", "Could not load ONNX backend from %s: %s", path, dlerror());
        return REDISMODULE_ERR;
    }
    RAI_LoadedBackend backend = {0};

    int (*init_backend)(int (*)(const char *, void **));
    init_backend =
        (int (*)(int (*)(const char *, void **)))(unsigned long)dlsym(handle, "RAI_InitBackendORT");
    if (!_ValidateFuncExists(ctx, init_backend, "RAI_InitBackendORT", "ONNX", path)) {
        goto error;
    }
    // Here we use the input callback to export functions from Redis and RedisAI
    // to the backend, by setting the backend's function pointers to the
    // corresponding functions in Redis/RedisAI.
    init_backend(RAI_ExportFunc);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateORT");
    if (!_ValidateFuncExists(ctx, backend.model_create, "RAI_ModelCreateORT", "ONNX", path)) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeORT");
    if (!_ValidateFuncExists(ctx, backend.model_free, "RAI_ModelFreeORT", "ONNX", path)) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
        unsigned long)dlsym(handle, "RAI_ModelRunORT");
    if (!_ValidateFuncExists(ctx, backend.model_run, "RAI_ModelRunORT", "ONNX", path)) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeORT");
    if (!_ValidateFuncExists(ctx, backend.model_serialize, "RAI_ModelSerializeORT", "ONNX", path)) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionORT");
    if (!_ValidateFuncExists(ctx, backend.get_version, "RAI_GetBackendVersionORT", "ONNX", path)) {
        goto error;
    }

    backend.get_memory_info =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryInfoORT");
    if (!_ValidateFuncExists(ctx, backend.get_memory_info, "RAI_GetMemoryInfoORT", "ONNX", path)) {
        goto error;
    }

    backend.get_memory_access_num =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryAccessORT");
    if (!_ValidateFuncExists(ctx, backend.get_memory_access_num, "RAI_GetMemoryAccessORT", "ONNX",
                             path)) {
        goto error;
    }

    backend.stop_long_running_sessions_cb =
        (void (*)(RedisModuleCtx *, RedisModuleEvent, uint64_t, void *))(unsigned long)dlsym(
            handle, "RAI_EnforceTimeoutORT");
    if (!_ValidateFuncExists(ctx, backend.stop_long_running_sessions_cb, "RAI_EnforceTimeoutORT",
                             "ONNX", path)) {
        goto error;
    }

    backend.add_new_device_cb =
        (int (*)(const char *))(unsigned long)dlsym(handle, "RAI_AddNewDeviceORT");
    if (!_ValidateFuncExists(ctx, backend.add_new_device_cb, "RAI_AddNewDeviceORT", "ONNX", path)) {
        goto error;
    }

    backend.get_max_run_sessions =
        (size_t(*)(void))(unsigned long)dlsym(handle, "RAI_GetGlobalRunSessionsLenORT");
    if (!_ValidateFuncExists(ctx, backend.get_max_run_sessions, "RAI_GetGlobalRunSessionsLenORT",
                             "ONNX", path)) {
        goto error;
    }

    RedisModule_SubscribeToServerEvent(ctx, RedisModuleEvent_CronLoop,
                                       backend.stop_long_running_sessions_cb);
    RAI_backends.onnx = backend;
    RedisModule_Log(ctx, "notice", "ONNX backend loaded from %s", path);
    return REDISMODULE_OK;

error:
    dlclose(handle);
    return REDISMODULE_ERR;
}

int RAI_LoadBackend(RedisModuleCtx *ctx, int backend, const char *path) {
    RedisModuleString *fullpath;

    if (path[0] == '/') {
        fullpath = RedisModule_CreateString(ctx, path, strlen(path));
    } else {
        const char *backends_path = Config_GetBackendsPath();
        fullpath = RedisModule_CreateStringPrintf(ctx, "%s/%s", backends_path, path);
    }

    int ret;
    switch (backend) {
    case RAI_BACKEND_TENSORFLOW:
        ret = RAI_LoadBackend_TensorFlow(ctx, RedisModule_StringPtrLen(fullpath, NULL));
        break;
    case RAI_BACKEND_TFLITE:
        ret = RAI_LoadBackend_TFLite(ctx, RedisModule_StringPtrLen(fullpath, NULL));
        break;
    case RAI_BACKEND_TORCH:
        ret = RAI_LoadBackend_Torch(ctx, RedisModule_StringPtrLen(fullpath, NULL));
        break;
    case RAI_BACKEND_ONNXRUNTIME:
        ret = RAI_LoadBackend_ONNXRuntime(ctx, RedisModule_StringPtrLen(fullpath, NULL));
        break;
    }
    RedisModule_FreeString(ctx, fullpath);
    return ret;
}

int RAI_LoadDefaultBackend(RedisModuleCtx *ctx, int backend) {
    switch (backend) {
    case RAI_BACKEND_TENSORFLOW:
        return RAI_LoadBackend(ctx, backend, "redisai_tensorflow/redisai_tensorflow.so");
    case RAI_BACKEND_TFLITE:
        return RAI_LoadBackend(ctx, backend, "redisai_tflite/redisai_tflite.so");
    case RAI_BACKEND_TORCH:
        return RAI_LoadBackend(ctx, backend, "redisai_torch/redisai_torch.so");
    case RAI_BACKEND_ONNXRUNTIME:
        return RAI_LoadBackend(ctx, backend, "redisai_onnxruntime/redisai_onnxruntime.so");
    }

    return REDISMODULE_ERR;
}
