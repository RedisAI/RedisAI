#define REDISMODULE_MAIN
#define _GNU_SOURCE
#include <string.h>
#include "backends/util.h"
#include "backends/torch.h"
#include "backends/backends_api.h"
#include "util/arr.h"
#include "util/dictionaries.h"
#include "libtorch_c/torch_c.h"
#include "redis_ai_objects/script.h"
#include "redis_ai_objects/tensor.h"
#include "execution/execution_contexts/scriptRun_ctx.h"

int RAI_InitBackendTorch(int (*get_api_fn)(const char *, void *)) {

    // Export Redis callbacks.
    get_api_fn("RedisModule_Alloc", ((void **)&RedisModule_Alloc));
    get_api_fn("RedisModule_Calloc", ((void **)&RedisModule_Calloc));
    get_api_fn("RedisModule_Free", ((void **)&RedisModule_Free));
    get_api_fn("RedisModule_Realloc", ((void **)&RedisModule_Realloc));
    get_api_fn("RedisModule_Strdup", ((void **)&RedisModule_Strdup));
    get_api_fn("RedisModule_CreateString", ((void **)&RedisModule_CreateString));
    get_api_fn("RedisModule_FreeString", ((void **)&RedisModule_FreeString));
    get_api_fn("RedisModule_Call", ((void **)&RedisModule_Call));
    get_api_fn("RedisModule_CallReplyType", ((void **)&RedisModule_CallReplyType));
    get_api_fn("RedisModule_CallReplyStringPtr", ((void **)&RedisModule_CallReplyStringPtr));
    get_api_fn("RedisModule_CallReplyInteger", ((void **)&RedisModule_CallReplyInteger));
    get_api_fn("RedisModule_CallReplyLength", ((void **)&RedisModule_CallReplyLength));
    get_api_fn("RedisModule_CallReplyArrayElement", ((void **)&RedisModule_CallReplyArrayElement));
    get_api_fn("RedisModule_FreeCallReply", ((void **)&RedisModule_FreeCallReply));
    get_api_fn("RedisModule_GetThreadSafeContext", ((void **)&RedisModule_GetThreadSafeContext));
    get_api_fn("RedisModule_ThreadSafeContextLock", ((void **)&RedisModule_ThreadSafeContextLock));
    get_api_fn("RedisModule_ThreadSafeContextUnlock",
               ((void **)&RedisModule_ThreadSafeContextUnlock));
    get_api_fn("RedisModule_FreeThreadSafeContext", ((void **)&RedisModule_FreeThreadSafeContext));
    get_api_fn("RedisModule_StringPtrLen", ((void **)&RedisModule_StringPtrLen));

    // Export RedisAI callbacks.
    get_api_fn("RedisAI_InitError", ((void **)&RedisAI_InitError));
    get_api_fn("RedisAI_FreeError", ((void **)&RedisAI_FreeError));
    get_api_fn("RedisAI_GetError", ((void **)&RedisAI_GetError));
    get_api_fn("RedisAI_TensorCreateFromDLTensor", ((void **)&RedisAI_TensorCreateFromDLTensor));
    get_api_fn("RedisAI_TensorGetDLTensor", ((void **)&RedisAI_TensorGetDLTensor));
    get_api_fn("RedisAI_TensorGetShallowCopy", ((void **)&RedisAI_TensorGetShallowCopy));
    get_api_fn("RedisAI_TensorFree", ((void **)&RedisAI_TensorFree));
    get_api_fn("RedisAI_GetModelFromKeyspace", ((void **)&RedisAI_GetModelFromKeyspace));
    get_api_fn("RedisAI_ModelRunCtxCreate", ((void **)&RedisAI_ModelRunCtxCreate));
    get_api_fn("RedisAI_ModelRunCtxAddInput", ((void **)&RedisAI_ModelRunCtxAddInput));
    get_api_fn("RedisAI_ModelRunCtxNumOutputs", ((void **)&RedisAI_ModelRunCtxNumOutputs));
    get_api_fn("RedisAI_ModelRunCtxAddOutput", ((void **)&RedisAI_ModelRunCtxAddOutput));
    get_api_fn("RedisAI_ModelRunCtxOutputTensor", ((void **)&RedisAI_ModelRunCtxOutputTensor));
    get_api_fn("RedisAI_ModelRunCtxFree", ((void **)&RedisAI_ModelRunCtxFree));
    get_api_fn("RedisAI_ModelRun", ((void **)&RedisAI_ModelRun));

    return REDISMODULE_OK;
}

RAI_Model *RAI_ModelCreateTorch(RAI_Backend backend, const char *devicestr, RAI_ModelOpts opts,
                                const char *modeldef, size_t modellen, RAI_Error *error) {
    DLDeviceType dl_device;

    RAI_Device device = RAI_DEVICE_CPU;
    int64_t deviceid = 0;

    char **inputs_ = NULL;
    char **outputs_ = NULL;

    if (!parseDeviceStr(devicestr, &device, &deviceid)) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, "ERR unsupported device");
        return NULL;
    }

    switch (device) {
    case RAI_DEVICE_CPU:
        dl_device = kDLCPU;
        break;
    case RAI_DEVICE_GPU:
        dl_device = kDLCUDA;
        break;
    default:
        RAI_SetError(error, RAI_EMODELCONFIGURE, "ERR Error configuring model: unsupported device");
        return NULL;
    }

    char *error_descr = NULL;
    if (opts.backends_inter_op_parallelism > 0) {
        torchSetInterOpThreads(opts.backends_inter_op_parallelism, &error_descr);
    }

    if (error_descr != NULL) {
        RAI_SetError(error, RAI_EMODELCREATE, error_descr);
        RedisModule_Free(error_descr);
        return NULL;
    }

    if (opts.backends_intra_op_parallelism > 0) {
        torchSetIntraOpThreads(opts.backends_intra_op_parallelism, &error_descr);
    }
    if (error_descr) {
        RAI_SetError(error, RAI_EMODELCREATE, error_descr);
        RedisModule_Free(error_descr);
        return NULL;
    }

    void *model = torchLoadModel(modeldef, modellen, dl_device, deviceid, &error_descr);

    if (error_descr) {
        goto cleanup;
    }

    size_t ninputs = torchModelNumInputs(model, &error_descr);
    if (error_descr) {
        goto cleanup;
    }

    size_t noutputs = torchModelNumOutputs(model, &error_descr);
    if (error_descr) {
        goto cleanup;
    }

    inputs_ = array_new(char *, ninputs);
    outputs_ = array_new(char *, noutputs);

    for (size_t i = 0; i < ninputs; i++) {
        const char *input = torchModelInputNameAtIndex(model, i, &error_descr);
        if (error_descr) {
            goto cleanup;
        }
        inputs_ = array_append(inputs_, RedisModule_Strdup(input));
    }

    for (size_t i = 0; i < noutputs; i++) {
        const char *output = "";
        if (error_descr) {
            goto cleanup;
        }
        outputs_ = array_append(outputs_, RedisModule_Strdup(output));
    }

    char *buffer = RedisModule_Calloc(modellen, sizeof(*buffer));
    memcpy(buffer, modeldef, modellen);

    RAI_Model *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->model = model;
    ret->session = NULL;
    ret->backend = backend;
    ret->devicestr = RedisModule_Strdup(devicestr);
    ret->ninputs = ninputs;
    ret->inputs = inputs_;
    ret->noutputs = noutputs;
    ret->outputs = outputs_;
    ret->opts = opts;
    ret->refCount = 1;
    ret->data = buffer;
    ret->datalen = modellen;
    return ret;

cleanup:
    RAI_SetError(error, RAI_EMODELCREATE, error_descr);
    RedisModule_Free(error_descr);
    if (inputs_) {
        ninputs = array_len(inputs_);
        for (size_t i = 0; i < ninputs; i++) {
            RedisModule_Free(inputs_[i]);
        }
        array_free(inputs_);
    }
    if (outputs_) {
        noutputs = array_len(outputs_);
        for (size_t i = 0; i < noutputs; i++) {
            RedisModule_Free(outputs_[i]);
        }
        array_free(outputs_);
    }
    return NULL;
}

void RAI_ModelFreeTorch(RAI_Model *model, RAI_Error *error) {
    if (model->devicestr) {
        RedisModule_Free(model->devicestr);
    }
    if (model->data) {
        RedisModule_Free(model->data);
    }
    size_t ninputs = model->ninputs;
    for (size_t i = 0; i < ninputs; i++) {
        RedisModule_Free(model->inputs[i]);
    }
    array_free(model->inputs);

    size_t noutputs = model->noutputs;
    for (size_t i = 0; i < noutputs; i++) {
        RedisModule_Free(model->outputs[i]);
    }
    array_free(model->outputs);

    torchDeallocContext(model->model);
}

int RAI_ModelRunTorch(RAI_Model *model, RAI_ExecutionCtx **ectxs, RAI_Error *error) {
    const size_t nbatches = array_len(ectxs);
    if (nbatches == 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR No batches to run");
        return 1;
    }

    const size_t ninputs = RAI_ExecutionCtx_NumInputs(ectxs[0]);
    const size_t noutputs = RAI_ExecutionCtx_NumOutputs(ectxs[0]);

    RAI_Tensor *inputs[ninputs];

    DLManagedTensor *inputs_dl[ninputs];
    DLManagedTensor *outputs_dl[noutputs];

    size_t batch_sizes[nbatches];
    size_t batch_offsets[nbatches];
    size_t total_batch_size = 0;

    if (nbatches > 1) {
        if (ninputs > 0) {
            for (size_t b = 0; b < nbatches; ++b) {
                batch_sizes[b] = RAI_TensorDim(RAI_ExecutionCtx_GetInput(ectxs[b], 0), 0);
                total_batch_size += batch_sizes[b];
            }
            batch_offsets[0] = 0;
            for (size_t b = 1; b < nbatches; ++b) {
                batch_offsets[b] = batch_offsets[b - 1] + batch_sizes[b - 1];
            }
        }

        for (size_t i = 0; i < ninputs; ++i) {
            RAI_Tensor *batch[nbatches];

            for (size_t b = 0; b < nbatches; b++) {
                batch[b] = RAI_ExecutionCtx_GetInput(ectxs[b], i);
            }

            inputs[i] = RAI_TensorCreateByConcatenatingTensors(batch, nbatches);
            inputs_dl[i] = &inputs[i]->tensor;
        }
    } else {
        for (size_t i = 0; i < ninputs; ++i) {
            inputs[i] = RAI_TensorGetShallowCopy(RAI_ExecutionCtx_GetInput(ectxs[0], i));
            inputs_dl[i] = &inputs[i]->tensor;
        }
    }

    for (size_t i = 0; i < noutputs; ++i) {
        outputs_dl[i] = NULL;
    }

    char *error_descr = NULL;
    torchRunModel(RAI_ModelGetModel(model), ninputs, inputs_dl, noutputs, outputs_dl, &error_descr);

    for (size_t i = 0; i < ninputs; ++i) {
        RAI_TensorFree(inputs[i]);
    }

    if (error_descr != NULL) {
        RAI_SetError(error, RAI_EMODELRUN, error_descr);
        RedisModule_Free(error_descr);
        return 1;
    }

    for (size_t i = 0; i < noutputs; ++i) {
        if (outputs_dl[i] == NULL) {
            RAI_SetError(error, RAI_EMODELRUN,
                         "ERR Model did not generate the expected number of outputs");
            return 1;
        }
        RAI_Tensor *output_tensor = RAI_TensorCreateFromDLTensor(outputs_dl[i]);
        if (nbatches > 1) {
            if (outputs_dl[i]->dl_tensor.shape[0] != total_batch_size) {
                RAI_SetError(error, RAI_EMODELRUN,
                             "ERR Model did not generate the expected batch size");
                RAI_TensorFree(output_tensor);
                return 1;
            }
            for (size_t b = 0; b < nbatches; b++) {
                RAI_ExecutionCtx_SetOutput(ectxs[b],
                                           RAI_TensorCreateBySlicingTensor(
                                               output_tensor, batch_offsets[b], batch_sizes[b]),
                                           i);
            }
        } else {
            RAI_ExecutionCtx_SetOutput(ectxs[0], RAI_TensorGetShallowCopy(output_tensor), i);
        }
        RAI_TensorFree(output_tensor);
    }

    return 0;
}

int RAI_ModelSerializeTorch(RAI_Model *model, char **buffer, size_t *len, RAI_Error *error) {

    if (model->data) {
        *buffer = RedisModule_Calloc(model->datalen, sizeof(char));
        memcpy(*buffer, model->data, model->datalen);
        *len = model->datalen;
    } else {
        char *error_descr = NULL;
        torchSerializeModel(model->model, buffer, len, &error_descr);

        if (*buffer == NULL) {
            RAI_SetError(error, RAI_EMODELSERIALIZE, error_descr);
            RedisModule_Free(error_descr);
            return 1;
        }
    }

    return 0;
}

RAI_Script *RAI_ScriptCreateTorch(const char *devicestr, const char *scriptdef,
                                  const char **entryPoints, size_t nEntryPoints, RAI_Error *error) {
    DLDeviceType dl_device;

    RAI_Device device;
    int64_t deviceid;

    if (!parseDeviceStr(devicestr, &device, &deviceid)) {
        RAI_SetError(error, RAI_ESCRIPTCONFIGURE, "ERR unsupported device");
    }

    switch (device) {
    case RAI_DEVICE_CPU:
        dl_device = kDLCPU;
        break;
    case RAI_DEVICE_GPU:
        dl_device = kDLCUDA;
        break;
    default:
        RAI_SetError(error, RAI_ESCRIPTCONFIGURE,
                     "ERR Error configuring script: unsupported device");
        break;
    }

    char *error_descr = NULL;
    void *script = torchCompileScript(scriptdef, dl_device, deviceid, &error_descr);

    if (script == NULL) {
        RAI_SetError(error, RAI_ESCRIPTCREATE, error_descr);
        RedisModule_Free(error_descr);
        return NULL;
    }

    // Entry points validation
    for (size_t i = 0; i < nEntryPoints; i++) {
        const char *entryPoint = entryPoints[i];
        // Existence check.
        if (!torchScript_FunctionExists(script, entryPoint)) {
            torchDeallocContext(script);
            char *errMsg;
            asprintf(&errMsg, "Fuction %s does not exists in the given script.", entryPoint);
            RAI_SetError(error, RAI_ESCRIPTCREATE, errMsg);
            free(errMsg);
            return NULL;
        }

        // Check for number of arguments.
        size_t argCount = torchScript_FunctionArgumentCountByFunctionName(script, entryPoint);
        if (argCount != 3) {
            torchDeallocContext(script);
            char *errMsg;
            asprintf(&errMsg, "Wrong number of inputs in fuction %s. Expected 3 but was %ld",
                     entryPoint, argCount);
            RAI_SetError(error, RAI_ESCRIPTCREATE, errMsg);
            free(errMsg);
            return NULL;
        }

        // Check for legal arguments.
        TorchScriptFunctionArgumentType argType1 =
            torchScript_FunctionArgumentTypeByFunctionName(script, entryPoint, 0);
        TorchScriptFunctionArgumentType argType2 =
            torchScript_FunctionArgumentTypeByFunctionName(script, entryPoint, 1);
        TorchScriptFunctionArgumentType argType3 =
            torchScript_FunctionArgumentTypeByFunctionName(script, entryPoint, 2);
        if (argType1 != TENSOR_LIST || argType2 != STRING_LIST || argType3 != STRING_LIST) {
            torchDeallocContext(script);
            char *errMsg;
            asprintf(&errMsg,
                     "Wrong inputs type in fuction %s. Expected signature similar to: def "
                     "%s(tensors: List[Tensor], keys: List[str], args: List[str])",
                     entryPoint, entryPoint);
            RAI_SetError(error, RAI_ESCRIPTCREATE, errMsg);
            free(errMsg);
            return NULL;
        }
    }

    RAI_Script *ret = RedisModule_Calloc(1, sizeof(*ret));
    ret->script = script;
    ret->scriptdef = RedisModule_Strdup(scriptdef);
    ret->devicestr = RedisModule_Strdup(devicestr);
    ret->refCount = 1;
    ret->entryPointsDict = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
    for (size_t i = 0; i < nEntryPoints; i++) {
        AI_dictAdd(ret->entryPointsDict, (void *)entryPoints[i], NULL);
    }
    return ret;
}

void RAI_ScriptFreeTorch(RAI_Script *script, RAI_Error *error) {
    torchDeallocContext(script->script);
    AI_dictRelease(script->entryPointsDict);
    RedisModule_Free(script->scriptdef);
    RedisModule_Free(script->devicestr);
    RedisModule_Free(script);
}

int RAI_ScriptRunTorch(RAI_Script *script, const char *function, RAI_ExecutionCtx *ectx,
                       RAI_Error *error) {

    long nInputs = RAI_ExecutionCtx_NumInputs(ectx);
    long nOutputs = RAI_ExecutionCtx_NumOutputs(ectx);

    DLManagedTensor *inputs[nInputs];
    DLManagedTensor *outputs[nOutputs];

    for (size_t i = 0; i < nInputs; i++) {
        inputs[i] = &RAI_ExecutionCtx_GetInput(ectx, i)->tensor;
    }

    for (size_t i = 0; i < nOutputs; i++) {
        outputs[i] = &RAI_ExecutionCtx_GetOutput(ectx, i)->tensor;
    }

    char *error_descr = NULL;

    RAI_ScriptRunCtx *sctx = (RAI_ScriptRunCtx *)ectx;

    // Create inputs context on stack.
    TorchFunctionInputCtx inputsCtx = {.tensorInputs = inputs,
                                       .tensorCount = nInputs,
                                       .args = sctx->args,
                                       .argsCount = array_len(sctx->args),
                                       .keys = sctx->keys,
                                       .keysCount = array_len(sctx->keys)};

    torchRunScript(script->script, function, &inputsCtx, outputs, nOutputs, &error_descr);

    if (error_descr) {
        RAI_SetError(error, RAI_ESCRIPTRUN, error_descr);
        RedisModule_Free(error_descr);
        return 1;
    }

    for (size_t i = 0; i < nOutputs; i++) {
        RAI_ExecutionCtx_SetOutput(ectx, RAI_TensorCreateFromDLTensor(outputs[i]), i);
    }

    return 0;
}

const char *RAI_GetBackendVersionTorch(void) { return "NA"; }
