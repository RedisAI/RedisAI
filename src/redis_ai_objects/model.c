/**
 * model.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include <pthread.h>
#include "err.h"
#include "model.h"
#include "stats.h"
#include "version.h"
#include "model_struct.h"
#include "backends/util.h"
#include "backends/backends.h"
#include "rmutil/alloc.h"
#include "util/arr.h"
#include "util/dict.h"
#include "util/string_utils.h"

extern RedisModuleType *RedisAI_ModelType;

RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char *devicestr, RedisModuleString *tag,
                           RAI_ModelOpts opts, size_t ninputs, const char **inputs, size_t noutputs,
                           const char **outputs, const char *modeldef, size_t modellen,
                           RAI_Error *err) {
    RAI_Model *model;
    if (backend == RAI_BACKEND_TENSORFLOW) {
        if (!RAI_backends.tf.model_create_with_nodes) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
            return NULL;
        }
        model = RAI_backends.tf.model_create_with_nodes(backend, devicestr, opts, ninputs, inputs,
                                                        noutputs, outputs, modeldef, modellen, err);
    } else if (backend == RAI_BACKEND_TFLITE) {
        if (!RAI_backends.tflite.model_create) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
            return NULL;
        }
        model = RAI_backends.tflite.model_create(backend, devicestr, opts, modeldef, modellen, err);
    } else if (backend == RAI_BACKEND_TORCH) {
        if (!RAI_backends.torch.model_create) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
            return NULL;
        }
        model = RAI_backends.torch.model_create(backend, devicestr, opts, modeldef, modellen, err);
    } else if (backend == RAI_BACKEND_ONNXRUNTIME) {
        if (!RAI_backends.onnx.model_create) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
            return NULL;
        }
        model = RAI_backends.onnx.model_create(backend, devicestr, opts, modeldef, modellen, err);
    } else {
        RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
        return NULL;
    }

    if (model) {
        if (tag) {
            model->tag = RAI_HoldString(tag);
        } else {
            model->tag = RedisModule_CreateString(NULL, "", 0);
        }
    }

    return model;
}

void RAI_ModelFree(RAI_Model *model, RAI_Error *err) {
    if (__atomic_sub_fetch(&model->refCount, 1, __ATOMIC_RELAXED) > 0) {
        return;
    }

    if (model->backend == RAI_BACKEND_TENSORFLOW) {
        if (!RAI_backends.tf.model_free) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
            return;
        }
        RAI_backends.tf.model_free(model, err);
    } else if (model->backend == RAI_BACKEND_TFLITE) {
        if (!RAI_backends.tflite.model_free) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
            return;
        }
        RAI_backends.tflite.model_free(model, err);
    } else if (model->backend == RAI_BACKEND_TORCH) {
        if (!RAI_backends.torch.model_free) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
            return;
        }
        RAI_backends.torch.model_free(model, err);
    } else if (model->backend == RAI_BACKEND_ONNXRUNTIME) {
        if (!RAI_backends.onnx.model_free) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
            return;
        }
        RAI_backends.onnx.model_free(model, err);
    } else {
        RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend");
        return;
    }

    RedisModule_FreeString(NULL, model->tag);

    // If the run stats which is stored under this key is the same one that the model holds a
    // reference to, remove the entry from the global statistics dictionary as well. Otherwise,
    // this key has been overwritten - just release the old run stats.
    RAI_RunStats *stats = RAI_StatsGetEntry(model->info->key);
    if (stats == model->info) {
        RAI_StatsRemoveEntry(stats->key);
    }
    RAI_StatsFree(model->info);

    RedisModule_Free(model);
}

RAI_Model *RAI_ModelGetShallowCopy(RAI_Model *model) {
    __atomic_fetch_add(&model->refCount, 1, __ATOMIC_RELAXED);
    return model;
}

int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err) {
    int ret;

    switch (model->backend) {
    case RAI_BACKEND_TENSORFLOW:
        if (!RAI_backends.tf.model_serialize) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.tf.model_serialize(model, buffer, len, err);
        break;
    case RAI_BACKEND_TFLITE:
        if (!RAI_backends.tflite.model_serialize) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.tflite.model_serialize(model, buffer, len, err);
        break;
    case RAI_BACKEND_TORCH:
        if (!RAI_backends.torch.model_serialize) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.torch.model_serialize(model, buffer, len, err);
        break;
    case RAI_BACKEND_ONNXRUNTIME:
        if (!RAI_backends.onnx.model_serialize) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.onnx.model_serialize(model, buffer, len, err);
        break;
    default:
        RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
        return REDISMODULE_ERR;
    }

    return ret;
}

int RedisAI_ModelRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx, RedisModuleString **argv,
                                                      int argc) {
    RedisModule_KeyAtPos(ctx, 1);
    size_t startpos = 2;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    const char *str = RedisModule_StringPtrLen(argv[startpos], NULL);
    if (!strcasecmp(str, "TIMEOUT")) {
        startpos += 2;
    }
    startpos += 1;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    for (size_t argpos = startpos; argpos < argc; argpos++) {
        str = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(str, "OUTPUTS")) {
            continue;
        }
        RedisModule_KeyAtPos(ctx, argpos);
    }
    return REDISMODULE_OK;
}

int ModelExecute_ReportKeysPositions(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    if (argc < 4) {
        return REDISMODULE_ERR;
    }
    RedisModule_KeyAtPos(ctx, 1); // model key

    // Inputs num should be at position 3, after AI.MODELEXECUTE, <model_key>, "INPUTS".
    size_t arg_pos = 3;
    long long inputs_num;
    if (RedisModule_StringToLongLong(argv[arg_pos], &inputs_num) != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }

    // <output_count> is located two positions after the last input key (after "OUTPUTS")
    arg_pos += inputs_num + 2;
    if (argc <= arg_pos) {
        return REDISMODULE_ERR;
    }
    long long outputs_num;
    if (RedisModule_StringToLongLong(argv[arg_pos], &outputs_num) != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    if (argc <= arg_pos + outputs_num) {
        return REDISMODULE_ERR;
    }
    size_t first_input_pos = 4, first_output_pos = first_input_pos + inputs_num + 2;
    for (size_t i = first_input_pos; i < first_input_pos + inputs_num; i++) {
        RedisModule_KeyAtPos(ctx, i);
    }
    for (size_t i = first_output_pos; i < first_output_pos + outputs_num; i++) {
        RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
}

int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Model **model,
                             int mode, RAI_Error *err) {
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(key);
#ifndef REDISAI_LITE
        RedisModule_Log(ctx, "warning", "could not load %s from keyspace, key doesn't exist",
                        RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_EKEYEMPTY, "ERR model key is empty");
#else
        if (VerifyKeyInThisShard(ctx, keyName)) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EKEYEMPTY, "ERR model key is empty");
        } else {
            RAI_SetError(err, RAI_EKEYEMPTY,
                         "ERR CROSSSLOT Keys in request don't hash to the same slot");
        }
#endif
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(key) != RedisAI_ModelType) {
        RedisModule_CloseKey(key);
        RedisModule_Log(ctx, "warning", "%s is not a model",
                        RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_EMODELRUN, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *model = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}

inline size_t RAI_ModelGetNumInputs(RAI_Model *model) { return model->ninputs; }

inline size_t RAI_ModelGetNumOutputs(RAI_Model *model) { return model->noutputs; }

inline const char *RAI_ModelGetInputName(RAI_Model *model, size_t index) {
    return model->inputs[index];
}

const char *RAI_ModelGetOutputName(RAI_Model *model, size_t index) { return model->outputs[index]; }

inline void *RAI_ModelGetSession(RAI_Model *model) { return model->session; }

inline void *RAI_ModelGetModel(RAI_Model *model) { return model->model; }

RedisModuleType *RAI_ModelRedisType(void) { return RedisAI_ModelType; }
