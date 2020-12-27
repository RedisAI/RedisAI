
#include "modelRun_ctx.h"

static int _Model_RunCtxAddParam(RAI_ModelCtxParam **paramArr, const char *name,
                                 RAI_Tensor *tensor) {

    RAI_ModelCtxParam param = {
        .name = name,
        .tensor = tensor ? RAI_TensorGetShallowCopy(tensor) : NULL,
    };
    *paramArr = array_append(*paramArr, param);
    return 1;
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

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx *mctx, const char *inputName, RAI_Tensor *inputTensor) {
    return _Model_RunCtxAddParam(&mctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx *mctx, const char *outputName) {
    return _Model_RunCtxAddParam(&mctx->outputs, outputName, NULL);
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

void RAI_ModelRunCtxFree(RAI_ModelRunCtx *mctx) {
    for (size_t i = 0; i < array_len(mctx->inputs); ++i) {
        RAI_TensorFree(mctx->inputs[i].tensor);
    }

    for (size_t i = 0; i < array_len(mctx->outputs); ++i) {
        if (mctx->outputs[i].tensor) {
            RAI_TensorFree(mctx->outputs[i].tensor);
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
    if (strcasecmp(inputstr, "INPUTS") != 0) {
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
            if (RMAPI_FUNC_SUPPORTED(RedisModule_HoldString)) {
                RedisModule_HoldString(NULL, argv[argpos]);
            } else {
                RedisModule_RetainString(NULL, argv[argpos]);
            }
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
