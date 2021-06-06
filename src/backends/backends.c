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
#include <execution/background_workers.h>

#include "redismodule.h"

static bool _ValidateAPICreated(RedisModuleCtx *ctx, void *func_ptr, const char *func_name) {
    if (func_ptr == NULL) {
        RedisModule_Log(ctx, "warning", "Backend does not export %s", func_name);
        return false;
    }
    return true;
}

int RAI_GetApi(const char *func_name, void **targetPtrPtr) {

    if (strcmp("ThreadIdKey", func_name) == 0) {
        *targetPtrPtr = GetThreadId;
    } else if (strcmp("NumThreadsPerQueue", func_name) == 0) {
        *targetPtrPtr = GetNumThreadsPerQueue;
    } else if (strcmp("OnnxTimeout", func_name) == 0) {
        *targetPtrPtr = GetOnnxTimeout;
    } else {
        return RedisModule_GetApi(func_name, targetPtrPtr);
    }
    return REDISMODULE_OK;
}

RedisModuleString *RAI_GetModulePath(RedisModuleCtx *ctx) {
    Dl_info info;
    RedisModuleString *module_path = NULL;
    if (dladdr(RAI_GetModulePath, &info)) {
        char *dli_fname = RedisModule_Strdup(info.dli_fname);
        const char *dli_dirname = dirname(dli_fname);
        module_path = RedisModule_CreateString(ctx, dli_dirname, strlen(dli_dirname));
        RedisModule_Free(dli_fname);
    }

    return module_path;
}

RedisModuleString *RAI_GetBackendsPath(RedisModuleCtx *ctx) {
    Dl_info info;
    RedisModuleString *backends_path = NULL;
    if (RAI_BackendsPath != NULL) {
        backends_path = RedisModule_CreateString(ctx, RAI_BackendsPath, strlen(RAI_BackendsPath));
    } else {
        RedisModuleString *module_path = RAI_GetModulePath(ctx);
        backends_path = RedisModule_CreateStringPrintf(ctx, "%s/backends",
                                                       RedisModule_StringPtrLen(module_path, NULL));
        RedisModule_FreeString(ctx, module_path);
    }

    return backends_path;
}

const char *GetBackendName(RAI_Backend backend) {
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
    RAI_LoadedBackend backend = {0};   // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void *));
    init_backend =
        (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(handle, "RAI_InitBackendTF");
    if (!_ValidateAPICreated(ctx, init_backend, "RAI_InitBackendTF")) {
        goto error;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create_with_nodes =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, size_t, const char **, size_t,
                         const char **, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTF");
    if (!_ValidateAPICreated(ctx, backend.model_create_with_nodes, "RAI_ModelCreateTF")) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTF");
    if (!_ValidateAPICreated(ctx, backend.model_free, "RAI_ModelFreeTF")) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
      unsigned long)dlsym(handle, "RAI_ModelRunTF");
    if (!_ValidateAPICreated(ctx, backend.model_run, "RAI_ModelRunTF")) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        (unsigned long)dlsym(handle, "RAI_ModelSerializeTF"));
    if (!_ValidateAPICreated(ctx, backend.model_serialize, "RAI_ModelSerializeTF")) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTF");
    if (!_ValidateAPICreated(ctx, backend.get_version, "RAI_GetBackendVersionTF")) {
        goto error;
    }

    RAI_backends.tf = backend;
    RedisModule_Log(ctx, "notice", "TF backend loaded from %s", path);
    return REDISMODULE_OK;

    error:
    dlclose(handle);
    RedisModule_Log(ctx, "warning", "TF backend not loaded from %s", path);
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
    RAI_LoadedBackend backend = {0};   // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTFLite");
    if (!_ValidateAPICreated(ctx, init_backend, "RAI_InitBackendTFLite")) {
        goto error;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTFLite");
    if (!_ValidateAPICreated(ctx, backend.model_create, "RAI_ModelCreateTFLite")) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTFLite");
    if (!_ValidateAPICreated(ctx, backend.model_free, "RAI_ModelFreeTFLite")) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
      unsigned long)dlsym(handle, "RAI_ModelRunTFLite");
    if (!_ValidateAPICreated(ctx, backend.model_run, "RAI_ModelRunTFLite")) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTFLite");
    if (!_ValidateAPICreated(ctx, backend.model_serialize, "RAI_ModelSerializeTFLite")) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTFLite");
    if (!_ValidateAPICreated(ctx, backend.get_version, "RAI_GetBackendVersionTFLite")) {
        goto error;
    }

    RAI_backends.tflite = backend;
    RedisModule_Log(ctx, "notice", "TFLITE backend loaded from %s", path);
    return REDISMODULE_OK;

    error:
    dlclose(handle);
    RedisModule_Log(ctx, "warning", "TFLITE backend not loaded from %s", path);
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

    RAI_LoadedBackend backend = {0};  // Initialize all the callbacks to NULL.

    int (*init_backend)(int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTorch");
    if (!_ValidateAPICreated(ctx, init_backend, "RAI_InitBackendTorch")) {
        goto error;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTorch");
    if (!_ValidateAPICreated(ctx, backend.model_create, "RAI_ModelCreateTorch")) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTorch");
    if (!_ValidateAPICreated(ctx, backend.model_free, "RAI_ModelFreeTorch")) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
      unsigned long)dlsym(handle, "RAI_ModelRunTorch");
    if (!_ValidateAPICreated(ctx, backend.model_run, "RAI_ModelRunTorch")) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTorch");
    if (!_ValidateAPICreated(ctx, backend.model_serialize, "RAI_ModelSerializeTorch")) {
        goto error;
    }

    backend.script_create = (RAI_Script * (*)(const char *, const char *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ScriptCreateTorch");
    if (!_ValidateAPICreated(ctx, backend.script_create, "RAI_ScriptCreateTorch")) {
        goto error;
    }

    backend.script_free =
        (void (*)(RAI_Script *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ScriptFreeTorch");
    if (!_ValidateAPICreated(ctx, backend.script_free, "RAI_ScriptFreeTorch")) {
        goto error;
    }

    backend.script_run = (int (*)(RAI_Script *, const char *, RAI_ExecutionCtx *, RAI_Error *))(
      unsigned long)dlsym(handle, "RAI_ScriptRunTorch");
    if (!_ValidateAPICreated(ctx, backend.script_run, "RAI_ScriptRunTorch")) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTorch");
    if (!_ValidateAPICreated(ctx, backend.get_version, "RAI_GetBackendVersionTorch")) {
        goto error;
    }

    RAI_backends.torch = backend;
    RedisModule_Log(ctx, "notice", "TORCH backend loaded from %s", path);
    return REDISMODULE_OK;

    error:
    dlclose(handle);
    RedisModule_Log(ctx, "warning", "TORCH backend not loaded from %s", path);
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
    init_backend = (int (*) (int (*)(const char *, void **)))(
        unsigned long)dlsym(handle, "RAI_InitBackendORT");
    if (!_ValidateAPICreated(ctx, init_backend, "RAI_InitBackendORT")) {
        goto error;
    }
    init_backend(RAI_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateORT");
    if (!_ValidateAPICreated(ctx, backend.model_create, "RAI_ModelCreateORT")) {
        goto error;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeORT");
    if (!_ValidateAPICreated(ctx, backend.model_free, "RAI_ModelFreeORT")) {
        goto error;
    }

    backend.model_run = (int (*)(RAI_Model * model, RAI_ExecutionCtx * *ectxs, RAI_Error * error))(
      unsigned long)dlsym(handle, "RAI_ModelRunORT");
    if (!_ValidateAPICreated(ctx, backend.model_run, "RAI_ModelRunORT")) {
        goto error;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeORT");
    if (!_ValidateAPICreated(ctx, backend.model_serialize, "RAI_ModelSerializeORT")) {
        goto error;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionORT");
    if (!_ValidateAPICreated(ctx, backend.get_version, "RAI_GetBackendVersionORT")) {
        goto error;
    }

    backend.get_memory_info =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryInfoORT");
    if (!_ValidateAPICreated(ctx, backend.get_memory_info, "RAI_GetMemoryInfoORT")) {
        goto error;
    }

    backend.get_memory_access_num =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryAccessORT");
    if (!_ValidateAPICreated(ctx, backend.get_memory_access_num, "RAI_GetMemoryAccessORT")) {
        goto error;
    }

    backend.enforce_runtime_duration =
        (void (*)(RedisModuleCtx *, RedisModuleEvent, uint64_t, void *))(unsigned long)dlsym(
            handle, "RAI_EnforceTimeoutORT");
    if (!_ValidateAPICreated(ctx, backend.enforce_runtime_duration, "RAI_EnforceTimeoutORT")) {
        goto error;
    }

    backend.add_new_device =
        (int (*)(const char *))(unsigned long)dlsym(handle, "RAI_AddNewDeviceORT");
    if (!_ValidateAPICreated(ctx, backend.add_new_device, "RAI_AddNewDeviceORT")) {
        goto error;
    }

    RedisModule_SubscribeToServerEvent(ctx, RedisModuleEvent_CronLoop,
      backend.enforce_runtime_duration);
    RAI_backends.onnx = backend;
    RedisModule_Log(ctx, "notice", "ONNX backend loaded from %s", path);
    return REDISMODULE_OK;

    error:
    dlclose(handle);
    RedisModule_Log(ctx, "warning", "ONNX backend not loaded from %s", path);
    return REDISMODULE_ERR;
}

int RAI_LoadBackend(RedisModuleCtx *ctx, int backend, const char *path) {
    RedisModuleString *fullpath;

    if (path[0] == '/') {
        fullpath = RedisModule_CreateString(ctx, path, strlen(path));
    } else {
        RedisModuleString *backends_path = RAI_GetBackendsPath(ctx);
        fullpath = RedisModule_CreateStringPrintf(
            ctx, "%s/%s", RedisModule_StringPtrLen(backends_path, NULL), path);
        RedisModule_FreeString(ctx, backends_path);
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
