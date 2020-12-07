
#define REDISMODULE_MAIN

#include "../../src/redisai.h"
#include <errno.h>
#include <string.h>

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
	return REDISMODULE_OK;
}
