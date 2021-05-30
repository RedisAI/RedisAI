#define REDISAI_EXTERN 1
#include "DAG_utils.h"
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include "util/arr.h"

static void _InitRunCtx(RAI_RunCtx *run_ctx) {
    RedisAI_InitError(&run_ctx->error);
    run_ctx->outputs = array_new(RAI_Tensor *, 1);
    pthread_mutex_init(&run_ctx->lock, NULL);
    pthread_cond_init(&run_ctx->cond, NULL);
}

static void _FreeRunCtx(RAI_RunCtx *run_ctx) {
    RedisAI_FreeError(run_ctx->error);
    for (size_t i = 0; i < array_len(run_ctx->outputs); i++) {
        RedisAI_TensorFree(run_ctx->outputs[i]);
    }
    array_free(run_ctx->outputs);
    pthread_mutex_destroy(&run_ctx->lock);
    pthread_cond_destroy(&run_ctx->cond);
}

static size_t _RunCtxNumOutputs(RAI_RunCtx run_ctx) { return array_len(run_ctx.outputs); }

static void *_getFromKeySpace(RedisModuleCtx *ctx, const char *keyNameStr) {

    RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RedisModule_FreeString(ctx, keyRedisStr);
    void *value = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
    return value;
}

static bool _assertError(RAI_Error *err, int status, const char *error_msg) {
    if (status != REDISMODULE_ERR) {
        return false;
    }
    // return true only if err contains the specific error_msg
    return strcmp(RedisAI_GetError(err), error_msg) == 0;
}

static void _DAGFinishFuncError(RAI_OnFinishCtx *onFinishCtx, void *private_data) {
    REDISMODULE_NOT_USED(onFinishCtx);
    REDISMODULE_NOT_USED(private_data);
    // Do nothing, this callback should not be used...
    RedisModule_Assert(false);
}

static void _DAGFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

    RAI_RunCtx *run_ctx = (RAI_RunCtx *)private_data;
    if (RedisAI_DAGRunError(onFinishCtx)) {
        const RAI_Error *error = RedisAI_DAGGetError(onFinishCtx);
        RedisAI_CloneError(run_ctx->error, error);
        goto finish;
    }
    size_t n_outputs = RedisAI_DAGNumOutputs(onFinishCtx);
    for (size_t i = 0; i < n_outputs; i++) {
        RAI_Tensor *t = (RAI_Tensor *)RedisAI_DAGOutputTensor(onFinishCtx, i);
        RedisModule_Assert(t != NULL);
        run_ctx->outputs = array_append(run_ctx->outputs, RedisAI_TensorGetShallowCopy(t));
    }

    // Verify that we return NULL as output for an index out of range.
    RAI_Tensor *t = (RAI_Tensor *)RedisAI_DAGOutputTensor(onFinishCtx, n_outputs);
    RedisModule_Assert(t == NULL);

finish:
    pthread_mutex_lock(&run_ctx->lock);
    pthread_cond_signal(&run_ctx->cond);
    pthread_mutex_unlock(&run_ctx->lock);
}

int testLoadTensor(RedisModuleCtx *ctx) {
    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    int res = LLAPIMODULE_ERR;
    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    if (RedisAI_DAGLoadTensor(run_info, "input", t) != REDISMODULE_OK) {
        goto cleanup;
    }
    t = (RAI_Tensor *)_getFromKeySpace(ctx, "b{1}");

    // cannot load more than one tensor under the same name.
    if (RedisAI_DAGLoadTensor(run_info, "input", t) != REDISMODULE_ERR) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

cleanup:
    RedisAI_DAGFree(run_info);
    return res;
}

int testModelRunOpError(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_Error *err;
    RedisAI_InitError(&err);
    int res = LLAPIMODULE_ERR;
    // The model m{1} should exist in key space.
    RAI_Model *model = (RAI_Model *)_getFromKeySpace(ctx, "m{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateModelRunOp(model);
    RedisAI_DAGRunOpAddInput(op, "first_input");

    // This model expect for 2 inputs not 1.
    int status = RedisAI_DAGAddRunOp(run_info, op, err);
    if (!_assertError(err, status,
                      "Number of keys given as INPUTS does not match model definition")) {
        goto cleanup;
    }
    RedisAI_ClearError(err);
    RedisAI_DAGRunOpAddInput(op, "second_input");
    status = RedisAI_DAGAddRunOp(run_info, op, err);

    // We still get an error since the model expects for an output as well.
    if (!_assertError(err, status,
                      "Number of keys given as OUTPUTS does not match model definition")) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

cleanup:
    RedisAI_FreeError(err);
    RedisAI_DAGRunOpFree(op);
    RedisAI_DAGFree(run_info);
    return res;
}

int testEmptyDAGError(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_Error *err;
    RedisAI_InitError(&err);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGLoadTensor(run_info, "input", t);

    int status = RedisAI_DAGRun(run_info, _DAGFinishFuncError, NULL, err);
    if (!_assertError(err, status, "ERR DAG is empty")) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

cleanup:
    RedisAI_FreeError(err);
    RedisAI_DAGFree(run_info);
    return res;
}

int testKeysMismatchError(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_Error *err;
    RedisAI_InitError(&err);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGLoadTensor(run_info, "input", t);

    RedisAI_DAGAddTensorGet(run_info, "non existing tensor");
    int status = RedisAI_DAGRun(run_info, _DAGFinishFuncError, NULL, err);
    if (!_assertError(err, status, "ERR INPUT key cannot be found in DAG")) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

cleanup:
    RedisAI_FreeError(err);
    RedisAI_DAGFree(run_info);
    return res;
}

int testBuildDAGFromString(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_RunCtx run_ctx;
    _InitRunCtx(&run_ctx);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGLoadTensor(run_info, "input1", t);

    const char *dag_string = "bad string";
    int status = RedisAI_DAGAddOpsFromString(run_info, dag_string, run_ctx.error);
    if (!_assertError(run_ctx.error, status, "DAG op should start with: '|>' ")) {
        goto cleanup;
    }
    RedisAI_ClearError(run_ctx.error);

    t = (RAI_Tensor *)_getFromKeySpace(ctx, "b{1}");
    RedisAI_DAGAddTensorSet(run_info, "input2", t);

    dag_string =
        "|> AI.MODELEXECUTE m{1} INPUTS 2 input1 input2 OUTPUTS 1 output |> bad_op no_tensor";
    status = RedisAI_DAGAddOpsFromString(run_info, dag_string, run_ctx.error);
    if (!_assertError(run_ctx.error, status, "Unsupported command within DAG")) {
        goto cleanup;
    }
    RedisAI_ClearError(run_ctx.error);
    RedisModule_Assert(RedisAI_DAGNumOps(run_info) == 1);

    dag_string =
        "|> AI.MODELEXECUTE m{1} INPUTS 2 input1 input2 OUTPUTS 1 output |> AI.TENSORGET output";
    if (RedisAI_DAGAddOpsFromString(run_info, dag_string, run_ctx.error) != REDISMODULE_OK) {
        goto cleanup;
    }
    RedisModule_Assert(RedisAI_DAGNumOps(run_info) == 3);
    RedisAI_DAGAddTensorGet(run_info, "input1");
    RedisModule_Assert(RedisAI_DAGNumOps(run_info) == 4);

    pthread_mutex_lock(&run_ctx.lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &run_ctx, run_ctx.error) != REDISMODULE_OK) {
        pthread_mutex_unlock(&run_ctx.lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&run_ctx.cond, &run_ctx.lock);
    pthread_mutex_unlock(&run_ctx.lock);

    // Verify that we received the expected tensor at the end of the run.
    RedisModule_Assert(_RunCtxNumOutputs(run_ctx) == 2);
    res = LLAPIMODULE_OK;

cleanup:
    _FreeRunCtx(&run_ctx);
    RedisAI_DAGFree(run_info);
    return res;
}

int testSimpleDAGRun(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_RunCtx run_ctx;
    _InitRunCtx(&run_ctx);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGLoadTensor(run_info, "input1", t);
    t = (RAI_Tensor *)_getFromKeySpace(ctx, "b{1}");
    RedisAI_DAGLoadTensor(run_info, "input2", t);

    // The model m{1} should exist in key space.
    RAI_Model *model = (RAI_Model *)_getFromKeySpace(ctx, "m{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateModelRunOp(model);
    RedisAI_DAGRunOpAddInput(op, "input1");
    RedisAI_DAGRunOpAddInput(op, "input2");
    RedisAI_DAGRunOpAddOutput(op, "output");
    if (RedisAI_DAGAddRunOp(run_info, op, run_ctx.error) != REDISMODULE_OK) {
        goto cleanup;
    }

    RedisAI_DAGAddTensorGet(run_info, "output");
    pthread_mutex_lock(&run_ctx.lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &run_ctx, run_ctx.error) != REDISMODULE_OK) {
        pthread_mutex_unlock(&run_ctx.lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&run_ctx.cond, &run_ctx.lock);
    pthread_mutex_unlock(&run_ctx.lock);

    // Verify that we received the expected tensor at the end of the run.
    RedisModule_Assert(_RunCtxNumOutputs(run_ctx) == 1);
    RAI_Tensor *out_tensor = run_ctx.outputs[0];
    double expceted[4] = {4, 9, 4, 9};
    double val;
    for (long long i = 0; i < 4; i++) {
        if (!RedisAI_TensorGetValueAsDouble(out_tensor, i, &val)) {
            goto cleanup;
        }
        if (expceted[i] != val) {
            goto cleanup;
        }
    }
    res = LLAPIMODULE_OK;

cleanup:
    _FreeRunCtx(&run_ctx);
    RedisAI_DAGFree(run_info);
    return res;
}

int testSimpleDAGRun2(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_RunCtx run_ctx;
    _InitRunCtx(&run_ctx);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *tensor = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGAddTensorSet(run_info, "input1", tensor);
    tensor = (RAI_Tensor *)_getFromKeySpace(ctx, "b{1}");
    RedisAI_DAGAddTensorSet(run_info, "input2", tensor);

    // The script myscript{1} should exist in key space.
    RAI_Script *script = (RAI_Script *)_getFromKeySpace(ctx, "myscript{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateScriptRunOp(script, "bar");
    RedisAI_DAGRunOpAddInput(op, "input1");
    RedisAI_DAGRunOpAddInput(op, "input2");
    RedisAI_DAGRunOpAddOutput(op, "output");
    if (RedisAI_DAGAddRunOp(run_info, op, run_ctx.error) != REDISMODULE_OK) {
        goto cleanup;
    }

    RedisAI_DAGAddTensorGet(run_info, "output");
    pthread_mutex_lock(&run_ctx.lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &run_ctx, run_ctx.error) != REDISMODULE_OK) {
        pthread_mutex_unlock(&run_ctx.lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&run_ctx.cond, &run_ctx.lock);
    pthread_mutex_unlock(&run_ctx.lock);

    // Verify that we received the expected tensor at the end of the run.
    RedisModule_Assert(_RunCtxNumOutputs(run_ctx) == 1);
    RAI_Tensor *out_tensor = run_ctx.outputs[0];
    double expceted[4] = {4, 6, 4, 6};
    double val;
    for (long long i = 0; i < 4; i++) {
        if (!RedisAI_TensorGetValueAsDouble(out_tensor, i, &val)) {
            goto cleanup;
        }
        if (expceted[i] != val) {
            goto cleanup;
        }
    }
    res = LLAPIMODULE_OK;

cleanup:
    _FreeRunCtx(&run_ctx);
    RedisAI_DAGFree(run_info);
    return res;
}

int testSimpleDAGRun2Error(RedisModuleCtx *ctx) {

    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_RunCtx run_ctx;
    _InitRunCtx(&run_ctx);
    int res = LLAPIMODULE_ERR;

    RAI_Tensor *tensor = (RAI_Tensor *)_getFromKeySpace(ctx, "a{1}");
    RedisAI_DAGAddTensorSet(run_info, "input1", tensor);

    // The script myscript{1} should exist in key space.
    RAI_Script *script = (RAI_Script *)_getFromKeySpace(ctx, "myscript{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateScriptRunOp(script, "no_function");
    RedisAI_DAGRunOpAddInput(op, "input1");
    RedisAI_DAGRunOpAddOutput(op, "output");
    if (RedisAI_DAGAddRunOp(run_info, op, run_ctx.error) != REDISMODULE_OK) {
        goto cleanup;
    }

    RedisAI_DAGAddTensorGet(run_info, "output");
    pthread_mutex_lock(&run_ctx.lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &run_ctx, run_ctx.error) != REDISMODULE_OK) {
        pthread_mutex_unlock(&run_ctx.lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&run_ctx.cond, &run_ctx.lock);
    pthread_mutex_unlock(&run_ctx.lock);

    // Verify that we received an error in SCRIPTRUN op.
    RedisModule_Assert(_RunCtxNumOutputs(run_ctx) == 0);
    if (RedisAI_GetErrorCode(run_ctx.error) != RedisAI_ErrorCode_ESCRIPTRUN) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

cleanup:
    _FreeRunCtx(&run_ctx);
    RedisAI_DAGFree(run_info);
    return res;
}

int testDAGResnet(RedisModuleCtx *ctx) {
    RAI_RunCtx run_ctx;
    _InitRunCtx(&run_ctx);
    int res = LLAPIMODULE_ERR;

    // Build the DAG with LOAD->SCRIPTRUN->MODELRUN->MODELRUN-SCRIPTRUN->SCRIPTRUN->TENSORGET
    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();
    RAI_Tensor *t = (RAI_Tensor *)_getFromKeySpace(ctx, "image:{1}");
    RedisAI_DAGLoadTensor(run_info, "image:{1}", t);

    RAI_Script *script = (RAI_Script *)_getFromKeySpace(ctx, "imagenet_script1:{1}");
    RAI_DAGRunOp *script_op = RedisAI_DAGCreateScriptRunOp(script, "pre_process_3ch");
    RedisAI_DAGRunOpAddInput(script_op, "image:{1}");
    RedisAI_DAGRunOpAddOutput(script_op, "tmp_key:{1}");
    RedisAI_DAGAddRunOp(run_info, script_op, run_ctx.error);

    RAI_Model *model = (RAI_Model *)_getFromKeySpace(ctx, "imagenet_model1:{1}");
    RAI_DAGRunOp *model_op = RedisAI_DAGCreateModelRunOp(model);
    RedisAI_DAGRunOpAddInput(model_op, "tmp_key:{1}");
    RedisAI_DAGRunOpAddOutput(model_op, "tmp_key2_0");
    if (RedisAI_DAGAddRunOp(run_info, model_op, run_ctx.error) != REDISMODULE_OK)
        goto cleanup;

    model = (RAI_Model *)_getFromKeySpace(ctx, "imagenet_model2:{1}");
    model_op = RedisAI_DAGCreateModelRunOp(model);
    RedisAI_DAGRunOpAddInput(model_op, "tmp_key:{1}");
    RedisAI_DAGRunOpAddOutput(model_op, "tmp_key2_1");
    if (RedisAI_DAGAddRunOp(run_info, model_op, run_ctx.error) != REDISMODULE_OK)
        goto cleanup;

    script_op = RedisAI_DAGCreateScriptRunOp(script, "ensemble");
    RedisAI_DAGRunOpAddInput(script_op, "tmp_key2_0");
    RedisAI_DAGRunOpAddInput(script_op, "tmp_key2_1");
    RedisAI_DAGRunOpAddOutput(script_op, "tmp_key_1:{1}");
    RedisAI_DAGAddRunOp(run_info, script_op, run_ctx.error);

    script_op = RedisAI_DAGCreateScriptRunOp(script, "post_process");
    RedisAI_DAGRunOpAddInput(script_op, "tmp_key_1:{1}");
    RedisAI_DAGRunOpAddOutput(script_op, "output:{1}");
    RedisAI_DAGAddRunOp(run_info, script_op, run_ctx.error);

    RedisAI_DAGAddTensorGet(run_info, "output:{1}");

    pthread_mutex_lock(&run_ctx.lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &run_ctx, run_ctx.error) != REDISMODULE_OK) {
        pthread_mutex_unlock(&run_ctx.lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&run_ctx.cond, &run_ctx.lock);
    pthread_mutex_unlock(&run_ctx.lock);

    // Verify that we received the expected output.
    RedisModule_Assert(_RunCtxNumOutputs(run_ctx) == 1);
    RAI_Tensor *out_tensor = run_ctx.outputs[0];
    long long val;
    if (!RedisAI_TensorGetValueAsLongLong(out_tensor, 0, &val))
        goto cleanup;
    if (0 <= val && val <= 1000) {
        res = LLAPIMODULE_OK;
    }

cleanup:
    _FreeRunCtx(&run_ctx);
    RedisAI_DAGFree(run_info);
    return res;
}
