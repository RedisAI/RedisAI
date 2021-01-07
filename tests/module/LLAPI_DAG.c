#include "../../src/redisai.h"
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include "../../src/util/arr.h"

#define LLAPIMODULE_OK 0
#define LLAPIMODULE_ERR 1

pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t global_cond = PTHREAD_COND_INITIALIZER;

static RAI_Model *_getModelFromKeySpace(RedisModuleCtx *ctx, const char *keyNameStr) {

    RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Model *model = RedisModule_ModuleTypeGetValue(key);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);
    return model;
}

static void _DAGFinishFuncError(RAI_OnFinishCtx *onFinishCtx, void *private_data) {
    //Do nothing, this callback should not be used...
    RedisModule_Assert(false);
}

static void _DAGFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

    RAI_Error *err;
    RedisAI_InitError(&err);
    //todo: Add API to extract errors...

    size_t n_outputs = RedisAI_DAGNumOutputs(onFinishCtx);
    RAI_Tensor **outputs = *(RAI_Tensor ***)private_data;
    for (size_t i = 0; i < n_outputs; i++) {
        RAI_Tensor *t = RedisAI_DAGOutputTensor(onFinishCtx, i);
        RedisModule_Assert(t != NULL);
        outputs = array_append(outputs, RedisAI_TensorGetShallowCopy(t));
    }

    RedisAI_FreeError(err);
    pthread_cond_signal(&global_cond);
}

static int _testLoadError(RAI_DAGRunCtx *run_info) {

    RAI_Error *err;
    RedisAI_InitError(&err);
    int status = RedisAI_DAGLoadTensor(run_info, "non_existing_tensor", err);
    if (strcmp(RedisAI_GetError(err), "ERR tensor key is empty") == 0) {
        RedisModule_Assert(status == REDISMODULE_ERR);
        RedisAI_FreeError(err);
        return LLAPIMODULE_OK;
    }
    RedisAI_FreeError(err);
    return LLAPIMODULE_ERR;
}

static int _testModelRunOpError(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info) {

    RAI_Error *err;
    RedisAI_InitError(&err);
    // The model m{1} should exist in key space.
    RAI_Model *model = _getModelFromKeySpace(ctx, "m{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateModelRunOp(run_info, model);
    RedisAI_DAGRunOpAddInput(op, "first_input");

    // This model expect for 2 inputs not 1.
    int status = RedisAI_DAGAddRunOp(run_info, op, err);
    if (strcmp(RedisAI_GetError(err), "Number of keys given as INPUTS does not match model definition") != 0) {
        RedisAI_FreeError(err);
        return LLAPIMODULE_ERR;
    }
    RedisAI_ClearError(err);
    RedisModule_Assert(status == REDISMODULE_ERR);
    RedisAI_DAGRunOpAddInput(op, "second_input");
    status = RedisAI_DAGAddRunOp(run_info, op, err);
    if (strcmp(RedisAI_GetError(err), "Number of keys given as OUTPUTS does not match model definition") != 0) {
        RedisAI_FreeError(err);
        return LLAPIMODULE_ERR;
    }
    RedisModule_Assert(status == REDISMODULE_ERR);
    RedisAI_FreeError(err);
    RedisAI_DAGRunOpFree(op);
    return LLAPIMODULE_OK;
}

static int _testEmptyDAGError(RAI_DAGRunCtx *run_info) {

    RAI_Error* err;
    RedisAI_InitError(&err);
    int res = LLAPIMODULE_ERR;

    int status = RedisAI_DAGLoadTensor(run_info, "a{1}", err);
    if(status != REDISMODULE_OK) {
        goto cleanup;
    }

    if(RedisAI_DAGRun(run_info, _DAGFinishFuncError, NULL, err) ==
       REDISMODULE_OK) {
        goto cleanup;
    }
    if(strcmp(RedisAI_GetError(err), "ERR DAG is empty") != 0) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

    cleanup:
    RedisAI_FreeError(err);
    RedisAI_DAGFree(run_info);
    return res;
}

static int _testKeysMismatchError(RAI_DAGRunCtx *run_info) {

    RAI_Error* err;
    RedisAI_InitError(&err);
    int res = LLAPIMODULE_ERR;

    int status = RedisAI_DAGLoadTensor(run_info, "a{1}", err);
    if(status != REDISMODULE_OK) {
        goto cleanup;
    }

    RedisAI_DAGAddTensorGet(run_info, "non existing tensor", err);
    if(RedisAI_DAGRun(run_info, _DAGFinishFuncError, NULL, err) ==
       REDISMODULE_OK) {
        goto cleanup;
    }
    if(strcmp(RedisAI_GetError(err), "ERR INPUT key cannot be found in DAG") != 0) {
        goto cleanup;
    }
    res = LLAPIMODULE_OK;

    cleanup:
    RedisAI_FreeError(err);
    RedisAI_DAGFree(run_info);
    return res;
}

static int _testSimpleDAGRun(RedisModuleCtx *ctx, RAI_DAGRunCtx *run_info) {

    RAI_Error *err;
    RedisAI_InitError(&err);
    RAI_Tensor **outputs = array_new(RAI_Tensor *, 1);
    int res = LLAPIMODULE_ERR;

    int status = RedisAI_DAGLoadTensor(run_info, "a{1}", err);
    if (status != REDISMODULE_OK) {
        goto cleanup;
    }
    status = RedisAI_DAGLoadTensor(run_info, "b{1}", err);
    if (status != REDISMODULE_OK) {
        goto cleanup;
    }

    // The model m{1} should exist in key space.
    RAI_Model *model = _getModelFromKeySpace(ctx, "m{1}");
    RAI_DAGRunOp *op = RedisAI_DAGCreateModelRunOp(run_info, model);
    RedisAI_DAGRunOpAddInput(op, "a{1}");
    RedisAI_DAGRunOpAddInput(op, "b{1}");
    RedisAI_DAGRunOpAddOutput(op, "output");
    status = RedisAI_DAGAddRunOp(run_info, op, err);
    if (status != REDISMODULE_OK) {
        goto cleanup;
    }

    RedisAI_DAGAddTensorGet(run_info, "output", err);
    pthread_mutex_lock(&global_lock);
    if (RedisAI_DAGRun(run_info, _DAGFinishFunc, &outputs, err) != REDISMODULE_OK) {
        pthread_mutex_unlock(&global_lock);
        goto cleanup;
    }
    // Wait until the onFinish callback returns.
    pthread_cond_wait(&global_cond, &global_lock);

    // Verify that we received the expected tensor at the end of the run.
    RedisModule_Assert(array_len(outputs) == 1);
    RAI_Tensor *out_tensor = outputs[0];
    double expceted[4] = {4, 9, 4, 9};
    double val[4];
    for (long long i = 0; i < 4; i++) {
        if(RedisAI_TensorGetValueAsDouble(out_tensor, i, &val[i]) != 0) {
            goto cleanup;
        }
        if (expceted[i] != val[i]) {
            goto cleanup;
        }
    }
    RedisAI_TensorFree(out_tensor);
    res = LLAPIMODULE_OK;

    cleanup:
    RedisAI_FreeError(err);
    array_free(outputs);
    RedisAI_DAGFree(run_info);
    return res;
}

int RAI_llapi_DAGRun(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if(argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();

    // Test the case of a failure due to non existing tensor to load.
    if(_testLoadError(run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "LOAD error test failed");
    }
    // Test the case of a failure due to addition of a non compatible MODELRUN op.
    if(_testModelRunOpError(ctx, run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "MODELRUN op error test failed");
    }
    // Test the case of a failure due an empty DAG.
    if(_testEmptyDAGError(run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of a failure due to an op within a DAG whose inkey does not exist in the DAG.
    if(_testKeysMismatchError(run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of building and running a DAG with LOAD, TENSORGET and MODELRUN ops.
    if(_testSimpleDAGRun(ctx, run_info) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run test failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "DAG run success");
}
