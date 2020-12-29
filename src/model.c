/**
 * model.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include "model.h"
#include "version.h"
#include "backends.h"
#include "backends/util.h"
#include "model_struct.h"
#include "rmutil/alloc.h"
#include "run_info.h"
#include "stats.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include <pthread.h>
#include "DAG/dag.h"

RedisModuleType *RedisAI_ModelType = NULL;

static void *RAI_Model_RdbLoad(struct RedisModuleIO *io, int encver) {
    // if (encver != RAI_ENC_VER) {
    //   /* We should actually log an error here, or try to implement
    //      the ability to load older versions of our data structure. */
    //   return NULL;
    // }

    RAI_Backend backend = RedisModule_LoadUnsigned(io);
    const char *devicestr = RedisModule_LoadStringBuffer(io, NULL);

    RedisModuleString *tag = RedisModule_LoadString(io);

    const size_t batchsize = RedisModule_LoadUnsigned(io);
    const size_t minbatchsize = RedisModule_LoadUnsigned(io);

    const size_t ninputs = RedisModule_LoadUnsigned(io);
    const char **inputs = RedisModule_Alloc(ninputs * sizeof(char *));

    for (size_t i = 0; i < ninputs; i++) {
        inputs[i] = RedisModule_LoadStringBuffer(io, NULL);
    }

    const size_t noutputs = RedisModule_LoadUnsigned(io);

    const char **outputs = RedisModule_Alloc(ninputs * sizeof(char *));

    for (size_t i = 0; i < noutputs; i++) {
        outputs[i] = RedisModule_LoadStringBuffer(io, NULL);
    }

    RAI_ModelOpts opts = {
        .batchsize = batchsize,
        .minbatchsize = minbatchsize,
        .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
        .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
    };

    size_t len;
    char *buffer = NULL;

    if (encver <= 100) {
        buffer = RedisModule_LoadStringBuffer(io, &len);
    } else {
        len = RedisModule_LoadUnsigned(io);
        buffer = RedisModule_Alloc(len);
        const size_t n_chunks = RedisModule_LoadUnsigned(io);
        long long chunk_offset = 0;
        for (size_t i = 0; i < n_chunks; i++) {
            size_t chunk_len;
            char *chunk_buffer = RedisModule_LoadStringBuffer(io, &chunk_len);
            memcpy(buffer + chunk_offset, chunk_buffer, chunk_len);
            chunk_offset += chunk_len;
            RedisModule_Free(chunk_buffer);
        }
    }

    RAI_Error err = {0};

    RAI_Model *model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs,
                                       outputs, buffer, len, &err);

    if (err.code == RAI_EBACKENDNOTLOADED) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        int ret = RAI_LoadDefaultBackend(ctx, backend);
        if (ret == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "error", "Could not load default backend");
            RAI_ClearError(&err);
            return NULL;
        }
        RAI_ClearError(&err);
        model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs,
                                buffer, len, &err);
    }

    if (err.code != RAI_OK) {
        RedisModuleCtx *ctx = RedisModule_GetContextFromIO(io);
        RedisModule_Log(ctx, "error", "%s", err.detail);
        RAI_ClearError(&err);
        if (buffer) {
            RedisModule_Free(buffer);
        }
        return NULL;
    }

    RedisModule_Free(inputs);
    RedisModule_Free(outputs);
    RedisModule_Free(buffer);

    RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
    RedisModuleString *stats_keystr =
        RedisModule_CreateStringFromString(stats_ctx, RedisModule_GetKeyNameFromIO(io));
    const char *stats_devicestr = RedisModule_Strdup(devicestr);
    RedisModuleString *stats_tag = RAI_HoldString(NULL, tag);

    model->infokey =
        RAI_AddStatsEntry(stats_ctx, stats_keystr, RAI_MODEL, backend, stats_devicestr, stats_tag);

    RedisModule_FreeString(NULL, stats_keystr);

    return model;
}

static void RAI_Model_RdbSave(RedisModuleIO *io, void *value) {
    RAI_Model *model = (RAI_Model *)value;
    char *buffer = NULL;
    size_t len = 0;
    RAI_Error err = {0};

    int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

    if (err.code != RAI_OK) {
        RedisModuleCtx *stats_ctx = RedisModule_GetContextFromIO(io);
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
        if (buffer) {
            RedisModule_Free(buffer);
        }
        return;
    }

    RedisModule_SaveUnsigned(io, model->backend);
    RedisModule_SaveStringBuffer(io, model->devicestr, strlen(model->devicestr) + 1);
    RedisModule_SaveString(io, model->tag);
    RedisModule_SaveUnsigned(io, model->opts.batchsize);
    RedisModule_SaveUnsigned(io, model->opts.minbatchsize);
    RedisModule_SaveUnsigned(io, model->ninputs);
    for (size_t i = 0; i < model->ninputs; i++) {
        RedisModule_SaveStringBuffer(io, model->inputs[i], strlen(model->inputs[i]) + 1);
    }
    RedisModule_SaveUnsigned(io, model->noutputs);
    for (size_t i = 0; i < model->noutputs; i++) {
        RedisModule_SaveStringBuffer(io, model->outputs[i], strlen(model->outputs[i]) + 1);
    }
    long long chunk_size = getModelChunkSize();
    const size_t n_chunks = len / chunk_size + 1;
    RedisModule_SaveUnsigned(io, len);
    RedisModule_SaveUnsigned(io, n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
        RedisModule_SaveStringBuffer(io, buffer + i * chunk_size, chunk_len);
    }

    if (buffer) {
        RedisModule_Free(buffer);
    }
}

static void RAI_Model_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_Model *model = (RAI_Model *)value;

    char *buffer = NULL;
    size_t len = 0;
    RAI_Error err = {0};

    int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

    if (err.code != RAI_OK) {

        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
        if (buffer) {
            RedisModule_Free(buffer);
        }
        return;
    }

    // AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1
    // name2 ...] model_blob

    RedisModuleString **inputs_ = array_new(RedisModuleString *, model->ninputs);
    RedisModuleString **outputs_ = array_new(RedisModuleString *, model->noutputs);

    RedisModuleCtx *ctx = RedisModule_GetContextFromIO(aof);

    for (size_t i = 0; i < model->ninputs; i++) {
        inputs_ = array_append(
            inputs_, RedisModule_CreateString(ctx, model->inputs[i], strlen(model->inputs[i])));
    }

    for (size_t i = 0; i < model->noutputs; i++) {
        outputs_ = array_append(
            outputs_, RedisModule_CreateString(ctx, model->outputs[i], strlen(model->outputs[i])));
    }

    long long chunk_size = getModelChunkSize();
    const size_t n_chunks = len / chunk_size + 1;
    RedisModuleString **buffers_ = array_new(RedisModuleString *, n_chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
        buffers_ = array_append(buffers_,
                                RedisModule_CreateString(ctx, buffer + i * chunk_size, chunk_len));
    }

    if (buffer) {
        RedisModule_Free(buffer);
    }

    const char *backendstr = RAI_BackendName(model->backend);

    RedisModule_EmitAOF(aof, "AI.MODELSET", "sccsclclcvcvcv", key, backendstr, model->devicestr,
                        model->tag, "BATCHSIZE", model->opts.batchsize, "MINBATCHSIZE",
                        model->opts.minbatchsize, "INPUTS", inputs_, model->ninputs, "OUTPUTS",
                        outputs_, model->noutputs, "BLOB", buffers_, n_chunks);

    for (size_t i = 0; i < model->ninputs; i++) {
        RedisModule_FreeString(ctx, inputs_[i]);
    }
    array_free(inputs_);

    for (size_t i = 0; i < model->noutputs; i++) {
        RedisModule_FreeString(ctx, outputs_[i]);
    }
    array_free(outputs_);

    for (size_t i = 0; i < n_chunks; i++) {
        RedisModule_FreeString(ctx, buffers_[i]);
    }
    array_free(buffers_);
}

/* Return REDISMODULE_ERR if there was an error getting the Model.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                             RAI_Model **model, int mode) {
    *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(*key);
        RedisModule_ReplyWithError(ctx, "ERR model key is empty");
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_ModelType) {
        RedisModule_CloseKey(*key);
        RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *model = RedisModule_ModuleTypeGetValue(*key);
    RedisModule_CloseKey(*key);
    return REDISMODULE_OK;
}

// TODO: pass err in?
static void RAI_Model_DTFree(void *value) {
    RAI_Error err = {0};
    RAI_ModelFree(value, &err);
    if (err.code != RAI_OK) {
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
    }
}

int RAI_ModelInit(RedisModuleCtx *ctx) {
    RedisModuleTypeMethods tmModel = {.version = REDISMODULE_TYPE_METHOD_VERSION,
                                      .rdb_load = RAI_Model_RdbLoad,
                                      .rdb_save = RAI_Model_RdbSave,
                                      .aof_rewrite = RAI_Model_AofRewrite,
                                      .mem_usage = NULL,
                                      .free = RAI_Model_DTFree,
                                      .digest = NULL};

    RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", RAI_ENC_VER_MM, &tmModel);
    return RedisAI_ModelType != NULL;
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
        model->ninputs = ninputs;
        model->noutputs = noutputs;
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

RedisModuleType *RAI_ModelRedisType(void) { return RedisAI_ModelType; }

int RAI_ModelRunAsync(RAI_ModelRunCtx *mctx, RAI_OnFinishCB ModelAsyncFinish, void *private_data) {

    RedisAI_RunInfo *rinfo = NULL;
    if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    rinfo->single_op_dag = 1;
    rinfo->OnFinish = (RedisAI_OnFinishCB)ModelAsyncFinish;
    rinfo->private_data = private_data;

    RAI_DagOp *op;
    if (RAI_InitDagOp(&op) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    op->commandType = REDISAI_DAG_CMD_MODELRUN;
    Dag_PopulateOp(op, mctx, NULL, NULL, NULL);

    rinfo->dagOps = array_append(rinfo->dagOps, op);
    rinfo->dagOpCount = 1;
    return DAG_InsertDAGToQueue(rinfo);
}
