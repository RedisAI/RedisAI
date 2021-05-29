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
#include "execution/onnx_timeout.h"

#include "redismodule.h"

int RAI_GetApi(const char *func_name, void **targetPtrPtr) {

    if (strcmp("ThreadIdKey", func_name) == 0) {
        *targetPtrPtr = GetQueueThreadIdKey;
    } else if (strcmp("NumThreadsPerQueue", func_name) == 0) {
        *targetPtrPtr = GetNumThreadsPerQueue;
    } else {
        return REDISMODULE_ERR;
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

const char *RAI_BackendName(int backend) {
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

    RAI_LoadedBackend backend = {0};

    int (*init_backend)(int (*)(const char *, void *));
    init_backend =
        (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(handle, "RAI_InitBackendTF");
    if (init_backend == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_InitBackendTF. TF backend not "
                        "loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create_with_nodes =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, size_t, const char **, size_t,
                         const char **, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTF");
    if (backend.model_create_with_nodes == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelCreateTF. TF backend not "
                        "loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTF");
    if (backend.model_free == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelFreeTF. TF backend not "
                        "loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_run =
        (int (*)(RAI_ModelRunCtx **, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelRunTF");
    if (backend.model_run == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelRunTF. TF backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        (unsigned long)dlsym(handle, "RAI_ModelSerializeTF"));
    if (backend.model_serialize == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelSerializeTF. TF backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTF");
    if (backend.get_version == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetBackendVersionTF. TF backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    RAI_backends.tf = backend;

    RedisModule_Log(ctx, "notice", "TF backend loaded from %s", path);

    return REDISMODULE_OK;
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

    RAI_LoadedBackend backend = {0};

    int (*init_backend)(int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTFLite");
    if (init_backend == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_InitBackendTFLite. TFLITE "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTFLite");
    if (backend.model_create == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelCreateTFLite. TFLITE "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTFLite");
    if (backend.model_free == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelFreeTFLite. TFLITE "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_run = (int (*)(RAI_ModelRunCtx **, RAI_Error *))(unsigned long)dlsym(
        handle, "RAI_ModelRunTFLite");
    if (backend.model_run == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelRunTFLite. TFLITE "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTFLite");
    if (backend.model_serialize == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelSerializeTFLite. TFLITE "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTFLite");
    if (backend.get_version == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetBackendVersionTFLite. TFLite backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    RAI_backends.tflite = backend;

    RedisModule_Log(ctx, "notice", "TFLITE backend loaded from %s", path);

    return REDISMODULE_OK;
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

    RAI_LoadedBackend backend = {0};

    int (*init_backend)(int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *)))(unsigned long)dlsym(
        handle, "RAI_InitBackendTorch");
    if (init_backend == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_InitBackendTorch. TORCH "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }
    init_backend(RedisModule_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateTorch");
    if (backend.model_create == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelCreateTorch. TORCH "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeTorch");
    if (backend.model_free == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelFreeTorch. TORCH backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_run =
        (int (*)(RAI_ModelRunCtx **, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelRunTorch");
    if (backend.model_run == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelRunTorch. TORCH backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeTorch");
    if (backend.model_serialize == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelSerializeTorch. TORCH "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.script_create = (RAI_Script * (*)(const char *, const char *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ScriptCreateTorch");
    if (backend.script_create == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ScriptCreateTorch. TORCH "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.script_free =
        (void (*)(RAI_Script *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ScriptFreeTorch");
    if (backend.script_free == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ScriptFreeTorch. TORCH "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.script_run = (int (*)(RAI_ScriptRunCtx *, RAI_Error *))(unsigned long)dlsym(
        handle, "RAI_ScriptRunTorch");
    if (backend.script_run == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ScriptRunTorch. TORCH backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionTorch");
    if (backend.get_version == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetBackendVersionTorch. TORCH backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    RAI_backends.torch = backend;

    RedisModule_Log(ctx, "notice", "TORCH backend loaded from %s", path);

    return REDISMODULE_OK;
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

    int (*init_backend)(int (*)(const char *, void *), int (*)(const char *, void *));
    init_backend = (int (*)(int (*)(const char *, void *), int (*)(const char *, void *)))(
        unsigned long)dlsym(handle, "RAI_InitBackendORT");
    if (init_backend == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_InitBackendORT. ONNX backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }
    init_backend(RedisModule_GetApi, (int (*)(const char *, void *))RAI_GetApi);

    backend.model_create =
        (RAI_Model * (*)(RAI_Backend, const char *, RAI_ModelOpts, const char *, size_t,
                         RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelCreateORT");
    if (backend.model_create == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelCreateORT. ONNX backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_free =
        (void (*)(RAI_Model *, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelFreeORT");
    if (backend.model_free == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelFreeORT. ONNX backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_run =
        (int (*)(RAI_ModelRunCtx **, RAI_Error *))(unsigned long)dlsym(handle, "RAI_ModelRunORT");
    if (backend.model_run == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelRunORT. ONNX backend not "
                        "loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.model_serialize = (int (*)(RAI_Model *, char **, size_t *, RAI_Error *))(
        unsigned long)dlsym(handle, "RAI_ModelSerializeORT");
    if (backend.model_serialize == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_ModelSerializeORT. ONNX "
                        "backend not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.get_version =
        (const char *(*)(void))(unsigned long)dlsym(handle, "RAI_GetBackendVersionORT");
    if (backend.get_version == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetBackendVersionORT. ONNX backend "
                        "not loaded from %s",
                        path);
        return REDISMODULE_ERR;
    }

    backend.get_memory_info =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryInfoORT");
    if (backend.get_memory_info == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetMemoryInfoORT. ONNX backend "
                        "not loaded from %s",
                        path);
    }
    backend.get_memory_access_num =
        (unsigned long long (*)(void))(unsigned long)dlsym(handle, "RAI_GetMemoryAccessORT");
    if (backend.get_memory_access_num == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export RAI_GetMemoryAccessORT. ONNX backend "
                        "not loaded from %s",
                        path);
    }

    backend.enforce_runtime_duration =
        (void (*)(RedisModuleCtx *, RedisModuleEvent, uint64_t, void *))(unsigned long)dlsym(
            handle, "OnnxEnforceTimeoutCallback");
    if (backend.enforce_runtime_duration == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export OnnxEnforceTimeoutCallback. ONNX backend "
                        "not loaded from %s",
                        path);
    }

    backend.add_new_device =
        (int (*)(const char *))(unsigned long)dlsym(handle, "AddDeviceToGlobalRunSessions");
    if (backend.add_new_device == NULL) {
        dlclose(handle);
        RedisModule_Log(ctx, "warning",
                        "Backend does not export AddDeviceToGlobalRunSessions. ONNX backend "
                        "not loaded from %s",
                        path);
    }

    RedisModule_SubscribeToServerEvent(ctx, RedisModuleEvent_CronLoop,
                                       backend.enforce_runtime_duration);

    RAI_backends.onnx = backend;
    RedisModule_Log(ctx, "notice", "ONNX backend loaded from %s", path);

    return REDISMODULE_OK;
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
