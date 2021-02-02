
#include "modelRun_ctx.h"
#include "util/string_utils.h"

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
