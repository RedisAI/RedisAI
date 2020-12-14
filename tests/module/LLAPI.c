
#define REDISMODULE_MAIN

#include "../../src/redisai.h"
#include <errno.h>
#include <string.h>
#include <pthread.h>

pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t global_cond = PTHREAD_COND_INITIALIZER;


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

	RAI_ModelRunCtx* mctx = RedisAI_GetModelRunCtx(onFinishCtx);
	RAI_Tensor *tensor = RedisAI_ModelRunCtxOutputTensor(mctx, 0);
	double expceted[4] = {4, 9, 4, 9};
	double val[4];
	for (long long i = 0; i < 4; i++) {
		if(RedisAI_TensorGetValueAsDouble(tensor, i, &val[i]) != 0) {
			pthread_cond_signal(&global_cond);
			return;
		}
		if (expceted[i] != val[i]) {
			pthread_cond_signal(&global_cond);
			return;
		}
	}
	*(int *)private_data = REDISMODULE_OK;
	pthread_cond_signal(&global_cond);
}

int RAI_llapi_modelRunAsync(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	REDISMODULE_NOT_USED(argv);

	if (argc>1) {
		RedisModule_WrongArity(ctx);
		return REDISMODULE_OK;
	}
	// The module m{1} exists in key space.
	const char *keyNameStr = "m{1}";
	RedisModuleString *keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
	RedisModuleKey *key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Model *model = RedisModule_ModuleTypeGetValue(key);
	RAI_ModelRunCtx* mctx = RedisAI_ModelRunCtxCreate(model);
	RedisModule_FreeString(ctx, keyRedisStr);
	RedisModule_CloseKey(key);

	// Load the tensors a{1} and b{1} and add them as inputs for m{1}.
	keyNameStr = "a{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
	key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Tensor *input1 = RedisModule_ModuleTypeGetValue(key);
	RedisAI_ModelRunCtxAddInput(mctx, "a", input1);

	keyNameStr = "b{1}";
	keyRedisStr = RedisModule_CreateString(ctx, keyNameStr, strlen(keyNameStr));
	key = RedisModule_OpenKey(ctx, keyRedisStr, REDISMODULE_READ);
	RAI_Tensor *input2 = RedisModule_ModuleTypeGetValue(key);
	RedisAI_ModelRunCtxAddInput(mctx, "b", input2);

	// Add c{1} as expected output tensor.
	RedisAI_ModelRunCtxAddOutput(mctx, "mul");
	int status = REDISMODULE_ERR;
	// Wait until the onFinish callback returns.
	pthread_mutex_lock(&global_lock);
	RedisAI_ModelRunAsync(mctx, ModelFinishFunc, &status);
	pthread_cond_wait(&global_cond, &global_lock);
	if (status == REDISMODULE_OK)
		return RedisModule_ReplyWithSimpleString(ctx, "Async Run Success");
	return RedisModule_ReplyWithSimpleString(ctx, "Async Run Failed");
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

	if(RedisModule_CreateCommand(ctx, "RAI_llapi.modelRunAsync", RAI_llapi_modelRunAsync, "",
	  0, 0, 0) == REDISMODULE_ERR)
		return REDISMODULE_ERR;
	return REDISMODULE_OK;
}
