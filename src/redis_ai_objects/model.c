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
#include "execution/run_info.h"
#include "execution/DAG/dag.h"

/* Return REDISMODULE_ERR if there was an error getting the Model.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Model **model,
                             int mode, RAI_Error *err) {
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(key);
        RAI_SetError(err, RAI_EKEYEMPTY, "ERR model key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(key) != RedisAI_ModelType) {
        RedisModule_CloseKey(key);
        RAI_SetError(err, RAI_EMODELRUN, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *model = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}

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
            model->tag = RAI_HoldString(NULL, tag);
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
    RAI_RemoveStatsEntry(model->infokey);

    RedisModule_Free(model);
}

int RAI_ModelRun(RAI_ModelRunCtx **mctxs, long long n, RAI_Error *err) {
    int ret;

    if (n == 0) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Nothing to run");
        return REDISMODULE_ERR;
    }

    RAI_ModelRunCtx **mctxs_arr = array_newlen(RAI_ModelRunCtx *, n);
    for (int i = 0; i < n; i++) {
        mctxs_arr[i] = mctxs[i];
    }

    switch (mctxs_arr[0]->model->backend) {
    case RAI_BACKEND_TENSORFLOW:
        if (!RAI_backends.tf.model_run) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TF");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.tf.model_run(mctxs_arr, err);
        break;
    case RAI_BACKEND_TFLITE:
        if (!RAI_backends.tflite.model_run) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TFLITE");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.tflite.model_run(mctxs_arr, err);
        break;
    case RAI_BACKEND_TORCH:
        if (!RAI_backends.torch.model_run) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.torch.model_run(mctxs_arr, err);
        break;
    case RAI_BACKEND_ONNXRUNTIME:
        if (!RAI_backends.onnx.model_run) {
            RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: ONNX");
            return REDISMODULE_ERR;
        }
        ret = RAI_backends.onnx.model_run(mctxs_arr, err);
        break;
    default:
        RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "ERR Unsupported backend");
        return REDISMODULE_ERR;
    }

    array_free(mctxs_arr);

    return ret;
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

    if (argc < 2) {
        return REDISMODULE_ERR;
    }
    RedisModule_KeyAtPos(ctx, 1); // model key

    // Inputs num should be the fourth arg.
    if (argc < 4) {
        return REDISMODULE_ERR;
    }
    long long inputs_num;
    if (RedisModule_StringToLongLong(argv[3], &inputs_num) != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }

    // The outputs num should be at the (inputs_num+6) position.
    if (argc < inputs_num + 6) {
        return REDISMODULE_ERR;
    }
    long long outputs_num;
    if (RedisModule_StringToLongLong(argv[inputs_num + 5], &outputs_num) != REDISMODULE_OK) {
        return REDISMODULE_ERR;
    }
    if (argc < inputs_num + outputs_num + 6) {
        return REDISMODULE_ERR;
    }
    for (size_t i = 4; i < 4 + inputs_num; i++) {
        RedisModule_KeyAtPos(ctx, i);
    }
    for (size_t i = 6 + inputs_num; i < 6 + inputs_num + outputs_num; i++) {
        RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
}

RedisModuleType *RAI_ModelRedisType(void) { return RedisAI_ModelType; }

size_t ModelGetNumInputs(RAI_Model *model) { return model->ninputs; }

size_t ModelGetNumOutputs(RAI_Model *model) { return model->noutputs; }

int RAI_ModelRunAsync(RAI_ModelRunCtx *mctx, RAI_OnFinishCB ModelAsyncFinish, void *private_data) {

    RedisAI_RunInfo *rinfo = NULL;
    RAI_InitRunInfo(&rinfo);

    rinfo->single_op_dag = 1;
    rinfo->OnFinish = (RedisAI_OnFinishCB)ModelAsyncFinish;
    rinfo->private_data = private_data;

    RAI_DagOp *op;
    RAI_InitDagOp(&op);
    op->commandType = REDISAI_DAG_CMD_MODELRUN;
    op->devicestr = mctx->model->devicestr;
    op->mctx = mctx;

    rinfo->dagOps = array_append(rinfo->dagOps, op);
    rinfo->dagOpCount = 1;
    if (DAG_InsertDAGToQueue(rinfo) != REDISMODULE_OK) {
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}
