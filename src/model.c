/**
 * model.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the RedisModuleType, and methods to manage
 * parsing and replying of tensor related commands or operations.
 *
 */

#include "model.h"
#include "backends.h"
#include "backends/util.h"
#include "model_struct.h"
#include "rmutil/alloc.h"
#include "run_info.h"
#include "stats.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include <pthread.h>

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
    return REDISMODULE_OK;
}

RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char *devicestr, const char *tag,
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
        model->tag = RedisModule_Strdup(tag);
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

    RedisModule_Free(model->tag);

    RAI_RemoveStatsEntry(model->infokey);

    RedisModule_Free(model);
}

RAI_ModelRunCtx *RAI_ModelRunCtxCreate(RAI_Model *model) {
#define PARAM_INITIAL_SIZE 10
    RAI_ModelRunCtx *mctx = RedisModule_Calloc(1, sizeof(*mctx));
    mctx->model = RAI_ModelGetShallowCopy(model);
    mctx->inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
    mctx->outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
    return mctx;
#undef PARAM_INITIAL_SIZE
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx *mctx, RAI_ModelCtxParam **paramArr,
                                const char *name, RAI_Tensor *tensor) {

    RAI_ModelCtxParam param = {
        .name = name,
        .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
    };
    *paramArr = array_append(*paramArr, param);
    return 1;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx *mctx, const char *inputName, RAI_Tensor *inputTensor) {
    return Model_RunCtxAddParam(mctx, &mctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx *mctx, const char *outputName) {
    return Model_RunCtxAddParam(mctx, &mctx->outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx *mctx) { return array_len(mctx->inputs); }

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx *mctx) { return array_len(mctx->outputs); }

RAI_Tensor *RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx *mctx, size_t index) {
    assert(RAI_ModelRunCtxNumInputs(mctx) > index && index >= 0);
    return mctx->inputs[index].tensor;
}

RAI_Tensor *RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx *mctx, size_t index) {
    assert(RAI_ModelRunCtxNumOutputs(mctx) > index && index >= 0);
    return mctx->outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx *mctx, int freeTensors) {
    if (freeTensors) {
        for (size_t i = 0; i < array_len(mctx->inputs); ++i) {
            RAI_TensorFree(mctx->inputs[i].tensor);
        }

        for (size_t i = 0; i < array_len(mctx->outputs); ++i) {
            if (mctx->outputs[i].tensor) {
                RAI_TensorFree(mctx->outputs[i].tensor);
            }
        }
    }

    array_free(mctx->inputs);
    array_free(mctx->outputs);

    RAI_Error err = {0};
    RAI_ModelFree(mctx->model, &err);

    if (err.code != RAI_OK) {
        // TODO: take it to client somehow
        RAI_ClearError(&err);
    }

    RedisModule_Free(mctx);
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

int RedisAI_Parse_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                        RAI_ModelRunCtx **mctx, RedisModuleString ***inkeys,
                                        RedisModuleString ***outkeys, RAI_Model **mto,
                                        RAI_Error *error) {
    if (argc < 3) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR wrong number of arguments for 'AI.MODELRUN' command");
        return -1;
    }

    const char *inputstr = RedisModule_StringPtrLen(argv[2], NULL);
    if (strcasecmp(inputstr, "INPUTS")) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
        return -1;
    }

    int is_input = 0;
    size_t ninputs = 0;
    size_t noutputs = 0;
    int outputs_flag_count = 0;
    size_t argpos = 3;

    for (; argpos <= argc - 1; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "OUTPUTS") && outputs_flag_count == 0) {
            is_input = 1;
            outputs_flag_count = 1;
        } else {
            RedisModule_RetainString(ctx, argv[argpos]);
            if (is_input == 0) {
                *inkeys = array_append(*inkeys, argv[argpos]);
                ninputs++;
            } else {
                *outkeys = array_append(*outkeys, argv[argpos]);
                noutputs++;
            }
        }
    }
    if ((*mto)->inputs && array_len((*mto)->inputs) != ninputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of names given as INPUTS during MODELSET and keys given as "
                     "INPUTS here do not match");
        return -1;
    }

    if ((*mto)->outputs && array_len((*mto)->outputs) != noutputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of names given as OUTPUTS during MODELSET and keys given as "
                     "OUTPUTS here do not match");
        return -1;
    }
    return argpos;
}

RedisModuleType *RAI_ModelRedisType(void) { return RedisAI_ModelType; }
