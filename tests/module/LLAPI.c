
#define REDISMODULE_MAIN
#define REDISAI_MAIN 1

#include "redisai.h"
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include "DAG_utils.h"

typedef enum LLAPI_status {LLAPI_RUN_NONE = 0,
						   LLAPI_RUN_SUCCESS,
						   LLAPI_RUN_ERROR,
						   LLAPI_NUM_OUTPUTS_ERROR
} LLAPI_status;

extern pthread_mutex_t global_lock;
extern pthread_cond_t global_cond;


int RAI_llapi_basic_check(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	REDISMODULE_NOT_USED(argv);

	if (argc>1) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}

	RAI_Error *err;
	RedisAI_InitError(&err);
	if(RedisAI_GetErrorCode(err) == RedisAI_ErrorCode_OK) {
        RedisModule_ReplyWithSimpleString(ctx, "OK");
    }
	RedisModule_ReplyWithError(ctx, "ERROR");
	RedisAI_FreeError(err);
	return REDISMODULE_OK;
}

static void _ScriptFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

	RAI_Error *err;
	if (RedisAI_InitError(&err) != REDISMODULE_OK) goto finish;
	RAI_ScriptRunCtx* sctx = RedisAI_GetAsScriptRunCtx(onFinishCtx, err);
	if(RedisAI_GetErrorCode(err) != RedisAI_ErrorCode_OK) {
		*(int *) private_data = LLAPI_RUN_ERROR;
		goto finish;
	}
	if(RedisAI_ScriptRunCtxNumOutputs(sctx) != 1) {
		*(int *) private_data = LLAPI_NUM_OUTPUTS_ERROR;
		goto finish;
	}
	RAI_Tensor *tensor = RedisAI_ScriptRunCtxOutputTensor(sctx, 0);
	double expceted[4] = {4, 6, 4, 6};
	double val[4];

	// Verify that we received the expected tensor at the end of the run.
	for (long long i = 0; i < 4; i++) {
		if(RedisAI_TensorGetValueAsDouble(tensor, i, &val[i]) != 0) {
			goto finish;
		}
		if (expceted[i] != val[i]) {
			goto finish;
		}
	}
	*(int *)private_data = LLAPI_RUN_SUCCESS;

	finish:
	RedisAI_FreeError(err);
	pthread_cond_signal(&global_cond);
}

static void _ModelFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

	RAI_Error *err;
	if (RedisAI_InitError(&err) != REDISMODULE_OK) goto finish;
	RAI_ModelRunCtx* mctx = RedisAI_GetAsModelRunCtx(onFinishCtx, err);
	if(RedisAI_GetErrorCode(err) != RedisAI_ErrorCode_OK) {
		*(int *) private_data = LLAPI_RUN_ERROR;
		goto finish;
	}
	if(RedisAI_ModelRunCtxNumOutputs(mctx) != 1) {
		*(int *) private_data = LLAPI_NUM_OUTPUTS_ERROR;
		goto finish;
	}
	RAI_Tensor *tensor = RedisAI_ModelRunCtxOutputTensor(mctx, 0);
	double expceted[4] = {4, 9, 4, 9};
	double val[4];

	// Verify that we received the expected tensor at the end of the run.
	for (long long i = 0; i < 4; i++) {
		if(RedisAI_TensorGetValueAsDouble(tensor, i, &val[i]) != 0) {
			goto finish;
		}
		if (expceted[i] != val[i]) {
			goto finish;
		}
	}
	*(int *)private_data = LLAPI_RUN_SUCCESS;

	finish:
	RedisAI_FreeError(err);
	pthread_cond_signal(&global_cond);
}

static int _ExecuteModelRunAsync(RedisModuleCtx *ctx, RAI_ModelRunCtx* mctx) {
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

static int _ExecuteScriptRunAsync(RedisModuleCtx *ctx, RAI_ScriptRunCtx* sctx) {
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

	if (argc>1) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}
	// The model m{1} should exist in key space.
	const char *keyNameStr = "m{1}";
	RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
	RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Model *model = RedisModule_ModuleTypeGetValue(key);
	RAI_ModelRunCtx* mctx = RedisAI_ModelRunCtxCreate(model);
	RedisModule_FreeString(ctx, keyRedisStr);
	RedisModule_CloseKey(key);

	// Test the case of a failure in the model run execution (no inputs specified).
	if(_ExecuteModelRunAsync(ctx, mctx) != LLAPI_RUN_ERROR) {
		return RedisModule_ReplyWithSimpleString(ctx, "Async run should end with an error");
	}

	mctx = RedisAI_ModelRunCtxCreate(model);
	// The tensors a{1} and b{1} should exist in key space.
	// Load the tensors a{1} and b{1} and add them as inputs for m{1}.
	keyNameStr = "a{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr,
	  strlen(keyNameStr));
	key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Tensor *input1 = RedisModule_ModuleTypeGetValue(key);
	RedisAI_ModelRunCtxAddInput(mctx, "a", input1);
	RedisModule_FreeString(ctx, keyRedisStr);
	RedisModule_CloseKey(key);

	keyNameStr = "b{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr,
	  strlen(keyNameStr));
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

	if (argc>1) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}
	// The script 'myscript{1}' should exist in key space.
	const char *keyNameStr = "myscript{1}";
	RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
	RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Script *script = RedisModule_ModuleTypeGetValue(key);
	RAI_ScriptRunCtx* sctx = RedisAI_ScriptRunCtxCreate(script, "bad_func");
	RedisModule_FreeString(ctx, keyRedisStr);
	RedisModule_CloseKey(key);

	// Test the case of a failure in the script run execution (func name does not exist in script).
	if(_ExecuteScriptRunAsync(ctx, sctx) != LLAPI_RUN_ERROR) {
		return RedisModule_ReplyWithSimpleString(ctx, "Async run should end with an error");
	}

	sctx = RedisAI_ScriptRunCtxCreate(script, "bar");
	RAI_Error *err;
	// The tensors a{1} and b{1} should exist in key space.
	// Load the tensors a{1} and b{1} and add them as inputs for the script.
	keyNameStr = "a{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr,
	  strlen(keyNameStr));
	key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Tensor *input1 = RedisModule_ModuleTypeGetValue(key);
	RedisAI_ScriptRunCtxAddInput(sctx, input1, err);
	RedisModule_FreeString(ctx, keyRedisStr);
	RedisModule_CloseKey(key);

	keyNameStr = "b{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr,
	  strlen(keyNameStr));
	key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Tensor *input2 = RedisModule_ModuleTypeGetValue(key);
	RedisAI_ScriptRunCtxAddInput(sctx, input2, err);
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

    if(argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    RAI_DAGRunCtx *run_info = RedisAI_DAGRunCtxCreate();

    // Test the case of a failure due to addition of a non compatible MODELRUN op.
    if(testModelRunOpError(ctx, run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "MODELRUN op error test failed");
    }
    // Test the case of a failure due an empty DAG.
    if(testEmptyDAGError(ctx, run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of a failure due to an op within a DAG whose inkey does not exist in the DAG.
    if(testKeysMismatchError(ctx, run_info) != LLAPIMODULE_OK) {
        RedisAI_DAGFree(run_info);
        return RedisModule_ReplyWithSimpleString(ctx, "DAG keys mismatch error test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of building and running a DAG with LOAD, TENSORGET and MODELRUN ops.
    if(testSimpleDAGRun(ctx, run_info) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of building and running a DAG with TENSORSET, SCRIPTRUN and TENSORGET ops.
    if(testSimpleDAGRun2(ctx, run_info) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run2 test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of building the same DAG as in previous test, but when this time it should return with an error.
    if(testSimpleDAGRun2Error(ctx, run_info) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Simple DAG run2 error test failed");
    }
    run_info = RedisAI_DAGRunCtxCreate();
    // Test the case of building DAG ops from string.
    if(testBuildDAGFromString(ctx, run_info) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "Build DAG from string test failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "DAG run success");
}

int RAI_llapi_DAG_resnet(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if(argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    if (testDAGResnet(ctx) != LLAPIMODULE_OK) {
        return RedisModule_ReplyWithSimpleString(ctx, "DAG resnet failed");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "DAG resnet success");
}


int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	REDISMODULE_NOT_USED(argv);
	REDISMODULE_NOT_USED(argc);

	if(RedisModule_Init(ctx, "RAI_llapi", 1, REDISMODULE_APIVER_1) ==
	   REDISMODULE_ERR)
		return REDISMODULE_ERR;

	if(RedisAI_Initialize(ctx) != REDISMODULE_OK)
		RedisModule_Log(ctx, "warning",
		  "could not initialize RedisAI api, running without AI support.");

	if(RedisModule_CreateCommand(ctx, "RAI_llapi.basic_check", RAI_llapi_basic_check, "",
	  0, 0, 0) == REDISMODULE_ERR)
		return REDISMODULE_ERR;

	if(RedisModule_CreateCommand(ctx, "RAI_llapi.modelRun", RAI_llapi_modelRun, "",
	  0, 0, 0) == REDISMODULE_ERR)
		return REDISMODULE_ERR;

	if(RedisModule_CreateCommand(ctx, "RAI_llapi.scriptRun", RAI_llapi_scriptRun, "",
	  0, 0, 0) == REDISMODULE_ERR)
		return REDISMODULE_ERR;


    if(RedisModule_CreateCommand(ctx, "RAI_llapi.DAGRun", RAI_llapi_DAGRun, "",
      0, 0, 0) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if(RedisModule_CreateCommand(ctx, "RAI_llapi.DAG_resnet", RAI_llapi_DAG_resnet, "",
      0, 0, 0) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

	return REDISMODULE_OK;
}
