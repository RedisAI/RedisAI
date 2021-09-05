
#define REDISMODULE_MAIN

#include "DAG_utils.h"
#include <errno.h>
#include <string.h>

typedef enum LLAPI_status {
    LLAPI_RUN_NONE = 0,
    LLAPI_RUN_SUCCESS,
    LLAPI_RUN_ERROR,
    LLAPI_NUM_OUTPUTS_ERROR
} LLAPI_status;

pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t global_cond = PTHREAD_COND_INITIALIZER;

int RAI_llapi_basic_check(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    RAI_Error *err;
    RedisAI_InitError(&err);
    if (RedisAI_GetErrorCode(err) == RedisAI_ErrorCode_OK) {
        RedisModule_ReplyWithSimpleString(ctx, "OK");
    }
    RedisModule_ReplyWithError(ctx, "ERROR");
    RedisAI_FreeError(err);
    return REDISMODULE_OK;
}

static void _ScriptFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

    RAI_Error *err;
    if (RedisAI_InitError(&err) != REDISMODULE_OK)
        goto finish;
    RAI_ScriptRunCtx *sctx = RedisAI_GetAsScriptRunCtx(onFinishCtx, err);
    if (RedisAI_GetErrorCode(err) != RedisAI_ErrorCode_OK) {
        *(int *)private_data = LLAPI_RUN_ERROR;
        goto finish;
    }
    if (RedisAI_ScriptRunCtxNumOutputs(sctx) != 1) {
        *(int *)private_data = LLAPI_NUM_OUTPUTS_ERROR;
        goto finish;
    }
    RAI_Tensor *tensor = RedisAI_ScriptRunCtxOutputTensor(sctx, 0);
    double expceted[4] = {4, 6, 4, 6};
    double val[4];

    // Verify that we received the expected tensor at the end of the run.
    for (long long i = 0; i < 4; i++) {
        if (!RedisAI_TensorGetValueAsDouble(tensor, i, &val[i])) {
            goto finish;
        }
        if (expceted[i] != val[i]) {
            goto finish;
        }
    }
    *(int *)private_data = LLAPI_RUN_SUCCESS;

finish:
    RedisAI_FreeError(err);
    pthread_mutex_lock(&global_lock);
    pthread_cond_signal(&global_cond);
    pthread_mutex_unlock(&global_lock);
}

static void _ModelFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

    RAI_Error *err;
    if (RedisAI_InitError(&err) != REDISMODULE_OK)
        goto finish;
    RAI_ModelRunCtx *mctx = RedisAI_GetAsModelRunCtx(onFinishCtx, err);
    if (RedisAI_GetErrorCode(err) != RedisAI_ErrorCode_OK) {
        *(int *)private_data = LLAPI_RUN_ERROR;
        goto finish;
    }
    if (RedisAI_ModelRunCtxNumOutputs(mctx) != 1) {
        *(int *)private_data = LLAPI_NUM_OUTPUTS_ERROR;
        goto finish;
    }
    RAI_Tensor *tensor = RedisAI_ModelRunCtxOutputTensor(mctx, 0);
    double expceted[4] = {4, 9, 4, 9};
    double val[4];

    // Verify that we received the expected tensor at the end of the run.
    for (long long i = 0; i < 4; i++) {
        if (!RedisAI_TensorGetValueAsDouble(tensor, i, &val[i])) {
            goto finish;
        }
        if (expceted[i] != val[i]) {
            goto finish;
        }
    }
    *(int *)private_data = LLAPI_RUN_SUCCESS;

finish:
    RedisAI_FreeError(err);
    pthread_mutex_lock(&global_lock);
    pthread_cond_signal(&global_cond);
    pthread_mutex_unlock(&global_lock);
}

static int _ExecuteModelRunAsync(RedisModuleCtx *ctx, RAI_ModelRunCtx *mctx) {
    LLAPI_status status = LLAPI_RUN_NONE;
    pthread_mutex_lock(&global_lock);
    if (RedisAI_ModelRunAsync(mctx, _ModelFinishFunc, &status) != REDISMODULE_OK) {
        pthread_mutex_unlock(&global_lock);
        RedisAI_ModelRunCtxFree(mctx);
        RedisModule_ReplyWithError(ctx, "Async run could not start");
        return LLAPI_RUN_NONE;
    }

    // Wait until the onFinish callback returns.
    pthread_cond_wait(&global_cond, &global_lock);
    pthread_mutex_unlock(&global_lock);
    RedisAI_ModelRunCtxFree(mctx);
    return status;
}

static int _ExecuteScriptRunAsync(RedisModuleCtx *ctx, RAI_ScriptRunCtx *sctx) {
    LLAPI_status status = LLAPI_RUN_NONE;
    pthread_mutex_lock(&global_lock);
    if (RedisAI_ScriptRunAsync(sctx, _ScriptFinishFunc, &status) != REDISMODULE_OK) {
        pthread_mutex_unlock(&global_lock);
        RedisAI_ScriptRunCtxFree(sctx);
        RedisModule_ReplyWithError(ctx, "Async run could not start");
        return LLAPI_RUN_NONE;
    }

    // Wait until the onFinish callback returns.
    pthread_cond_wait(&global_cond, &global_lock);
    pthread_mutex_unlock(&global_lock);
    RedisAI_ScriptRunCtxFree(sctx);
    return status;
}

int RAI_llapi_modelRun(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    // The model m{1} should exist in key space.
    const char *keyNameStr = "m{1}";
    RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Model *model = RedisModule_ModuleTypeGetValue(key);
    RAI_ModelRunCtx *mctx = RedisAI_ModelRunCtxCreate(model);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    // Test the case of a failure in the model run execution (no inputs specified).
    if (_ExecuteModelRunAsync(ctx, mctx) != LLAPI_RUN_ERROR) {
        return RedisModule_ReplyWithSimpleString(ctx, "Async run should end with an error");
    }

    mctx = RedisAI_ModelRunCtxCreate(model);
    // The tensors a{1} and b{1} should exist in key space.
    // Load the tensors a{1} and b{1} and add them as inputs for m{1}.
    keyNameStr = "a{1}";
    keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Tensor *input1 = RedisModule_ModuleTypeGetValue(key);
    RedisAI_ModelRunCtxAddInput(mctx, "a", input1);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    keyNameStr = "b{1}";
    keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Tensor *input2 = RedisModule_ModuleTypeGetValue(key);
    RedisAI_ModelRunCtxAddInput(mctx, "b", input2);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    // Add the expected output tensor.
    RedisAI_ModelRunCtxAddOutput(mctx, "mul");

    if (_ExecuteModelRunAsync(ctx, mctx) != LLAPI_RUN_SUCCESS)
        return RedisModule_ReplyWithSimpleString(ctx, "Async run failed");
    return RedisModule_ReplyWithSimpleString(ctx, "Async run success");
}

int RAI_llapi_scriptRun(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    // The script 'myscript{1}' should exist in key space.
    const char *keyNameStr = "myscript{1}";
    RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Script *script = RedisModule_ModuleTypeGetValue(key);
    RAI_ScriptRunCtx *sctx = RedisAI_ScriptRunCtxCreate(script, "bad_func");
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    // Test the case of a failure in the script run execution (func name does not exist in script).
    if (_ExecuteScriptRunAsync(ctx, sctx) != LLAPI_RUN_ERROR) {
        return RedisModule_ReplyWithSimpleString(ctx, "Async run should end with an error");
    }

    sctx = RedisAI_ScriptRunCtxCreate(script, "bar");
    RAI_Error *err;
    // The tensors a{1} and b{1} should exist in key space.
    // Load the tensors a{1} and b{1} and add them as inputs for the script.
    keyNameStr = "a{1}";
    keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Tensor *input1 = RedisModule_ModuleTypeGetValue(key);
    RedisAI_ScriptRunCtxAddTensorInput(sctx, input1);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    keyNameStr = "b{1}";
    keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
    key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
    RAI_Tensor *input2 = RedisModule_ModuleTypeGetValue(key);
    RedisAI_ScriptRunCtxAddTensorInput(sctx, input2);
    RedisModule_FreeString(ctx, keyRedisStr);
    RedisModule_CloseKey(key);

    // Add the expected output tensor.
    RedisAI_ScriptRunCtxAddOutput(sctx);

    if (_ExecuteScriptRunAsync(ctx, sctx) != LLAPI_RUN_SUCCESS)
        return RedisModule_ReplyWithSimpleString(ctx, "Async run failed");
    return RedisModule_ReplyWithSimpleString(ctx, "Async run success");
}

int RAI_llapi_DAGRun(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    // Test the case a successful and failure tensor load input to DAG.
    if (testLoadTensor(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "LOAD tensor test failed");
    }
    // Test the case of a failure due to addition of a non compatible MODELRUN op.
    if (testModelRunOpError(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "MODELRUN op error test failed");
    }
    // Test the case of a failure due an empty DAG.
    if (testEmptyDAGError(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    // Test the case of a failure due to an op within a DAG whose inkey does not exist in the DAG.
    if (testKeysMismatchError(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    // Test the case of building and running a DAG with LOAD, TENSORGET and MODELRUN ops.
    if (testSimpleDAGRun(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run test failed");
    }
    // Test the case of building and running a DAG with TENSORSET, SCRIPTRUN and TENSORGET ops.
    if (testSimpleDAGRun2(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run2 test failed");
    }
    // Test the case of building the same DAG as in previous test, but when this time it should
    // return with an error.
    if (testSimpleDAGRun2Error(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run2 error test failed");
    }
    // Test the case of building DAG ops from string.
    if (testBuildDAGFromString(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Build DAG from string test failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "DAG run success");
}

int RAI_llapi_DAG_resnet(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    if (testDAGResnet(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "DAG resnet failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "DAG resnet success");
}

int RAI_llapi_CreateTensor(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    int n_dims = 2;
    long long dims[] = {1, 4};
    // Try to create a tensor with a non-supported data type.
    RAI_Tensor *t = RedisAI_TensorCreate("INVALID", dims, n_dims);
    if (t != NULL) {
        return RedisModule_ReplyWithSimpleString(ctx, "invalid data type tensor create test failed");
    }

    // create an empty tensor and validate that in contains zeros
    t = RedisAI_TensorCreate("INT8", dims, n_dims);
    int8_t expected_blob[8] = {0};
    if (t == NULL || RedisAI_TensorLength(t) != dims[0] * dims[1] ||
        memcmp(RedisAI_TensorData(t), expected_blob, 8) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "empty tensor create test failed");
    }
    RedisAI_TensorFree(t);

    // This should fail since the blob contains only one null-terminated string, while the tensor's
    // len should be 4.
    RAI_Tensor *t1 = RedisAI_TensorCreate("STRING", dims, n_dims);
    const char *data_blob1 = "only one string\0";
    if (RedisAI_TensorSetData(t1, data_blob1, strlen(data_blob1)) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "invalid string tensor data set test failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "create tensor test success");
}


int RAI_llapi_ConcatenateTensors(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    int n_dims = 2;
    long long dims[] = {1, 4};

    // test concatenation of string tensors
    RAI_Tensor *t1 = RedisAI_TensorCreate("STRING", dims, n_dims);
    const char* data_blob1 = "first\0second\0third\0forth\0";
    size_t len_data_blob1 = 25;
    if (RedisAI_TensorSetData(t1, data_blob1, len_data_blob1) != 1) {
        return RedisModule_ReplyWithSimpleString(ctx, "string tensor data set test failed");
    }

    // the second tensor's shape is [2,4], while the previous shape was [1,4]
    dims[0] = 2;
    const char *data_blob2 = "A\0B\0C\0D\0E\0F\0G\0H\0";
    size_t len_data_blob2 = 16;
    RAI_Tensor *t2 = RedisAI_TensorCreate("STRING", dims, n_dims);
    if (RedisAI_TensorSetData(t2, data_blob2, len_data_blob2) != 1) {
        return RedisModule_ReplyWithSimpleString(ctx, "string tensor data set test failed");
    }

    RAI_Tensor *tensors[] = {t1, t2};
    RAI_Tensor *batched_tensor = RedisAI_TensorCreateByConcatenatingTensors(tensors, 2);
    RedisAI_TensorFree(t1);
    RedisAI_TensorFree(t2);
    const char *expected_batched_data = "first\0second\0third\0forth\0A\0B\0C\0D\0E\0F\0G\0H\0";
    size_t expected_batched_data_len = len_data_blob1 + len_data_blob2;
    if (batched_tensor == NULL || RedisAI_TensorNumDims(batched_tensor) != 2 ||
        RedisAI_TensorDim(batched_tensor, 0) != 3 || RedisAI_TensorDim(batched_tensor, 1) != 4 ||
        memcmp(expected_batched_data, RedisAI_TensorData(batched_tensor),
               expected_batched_data_len) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "string tensor concatenation test failed");
    }
    RedisAI_TensorFree(batched_tensor);
    return RedisModule_ReplyWithSimpleString(ctx, "concatenate tensors test success");
}

int RAI_llapi_SliceTensor(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    int n_dims = 2;
    long long dims[] = {3, 4};
    RAI_Tensor *batched_tensor = RedisAI_TensorCreate("STRING", dims, n_dims);
    const char *batched_data = "first\0second\0third\0forth\0A\0B\0C\0D\0E\0F\0G\0H\0";
    size_t len_data_batch1 = 25;
    size_t len_data_batch2 = 16;
    RedisAI_TensorSetData(batched_tensor, batched_data, len_data_batch1+len_data_batch2);

    // test slicing string tensors
    RAI_Tensor *t1 = RedisAI_TensorCreateBySlicingTensor(batched_tensor, 0, 1);
    RAI_Tensor *t2 = RedisAI_TensorCreateBySlicingTensor(batched_tensor, 1, 2);
    RedisAI_TensorFree(batched_tensor);

    if (t1 == NULL || RedisAI_TensorNumDims(t1) != 2 || RedisAI_TensorDim(t1, 0) != 1 ||
        RedisAI_TensorDim(t1, 1) != 4 ||
        memcmp(batched_data, RedisAI_TensorData(t1), len_data_batch1) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "string tensor slicing test failed");
    }

    if (t2 == NULL || RedisAI_TensorNumDims(t2) != 2 || RedisAI_TensorDim(t2, 0) != 2 ||
        RedisAI_TensorDim(t2, 1) != 4 ||
        memcmp(batched_data + len_data_batch1, RedisAI_TensorData(t2), len_data_batch2) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "string tensor slicing test failed");
    }
    RedisAI_TensorFree(t1);
    RedisAI_TensorFree(t2);
    return RedisModule_ReplyWithSimpleString(ctx, "slice tensors test success");
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if (RedisModule_Init(ctx, "RAI_llapi", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisAI_Initialize(ctx) != REDISMODULE_OK)
        RedisModule_Log(ctx, "warning",
                        "could not initialize RedisAI api, running without AI support.");

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.basic_check", RAI_llapi_basic_check, "", 0, 0,
                                  0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.modelRun", RAI_llapi_modelRun, "", 0, 0, 0) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.scriptRun", RAI_llapi_scriptRun, "", 0, 0, 0) ==
        REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.DAGRun", RAI_llapi_DAGRun, "", 0, 0, 0) ==
        REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.DAG_resnet", RAI_llapi_DAG_resnet, "", 0, 0, 0) ==
        REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.CreateTensor", RAI_llapi_CreateTensor, "", 0, 0,
                                  0) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.ConcatenateTensors", RAI_llapi_ConcatenateTensors, "", 0, 0,
                                  0) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "RAI_llapi.SliceTensor", RAI_llapi_SliceTensor, "", 0, 0,
                                  0) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
