
#define REDISMODULE_MAIN

#include "../../src/redisai.h"
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>

pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t global_cond = PTHREAD_COND_INITIALIZER;

typedef enum LLAPI_status {LLAPI_RUN_NONE = 0,
						   LLAPI_RUN_SUCCESS,
						   LLAPI_RUN_ERROR,
						   LLAPI_NUM_OUTPUTS_ERROR
} LLAPI_status;


int RAI_llapi_basic_check(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	REDISMODULE_NOT_USED(argv);

	if (argc>1) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}

	RAI_Error *err;
	RedisAI_InitError(&err);
	if(RedisAI_GetErrorCode(err) == RedisAI_ErrorCode_OK)
		return RedisModule_ReplyWithSimpleString(ctx, "OK");
	return RedisModule_ReplyWithError(ctx, "ERROR");
}

static void ModelFinishFunc(RAI_OnFinishCtx *onFinishCtx, void *private_data) {

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
	if (RedisAI_ModelRunAsync(mctx, ModelFinishFunc, &status) != REDISMODULE_OK) {
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
	return REDISMODULE_OK;
}
