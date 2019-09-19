#ifndef SRC_SCRIPT_H_
#define SRC_SCRIPT_H_

#include "config.h"
#include "script_struct.h"
#include "tensor.h"
#include "redismodule.h"
#include "err.h"

extern RedisModuleType *RedisAI_ScriptType;

int RAI_ScriptInit(RedisModuleCtx* ctx);
RAI_Script* RAI_ScriptCreate(RAI_Device device, int64_t deviceid, const char* devicestr, const char* scriptdef, RAI_Error* err);
void RAI_ScriptFree(RAI_Script* script, RAI_Error* err);

RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script, const char *fnname);
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, RAI_Tensor* inputTensor);
int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx);
size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx* sctx);
RAI_Tensor* RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx* sctx, size_t index);
void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx* sctx);

int RAI_ScriptRun(RAI_ScriptRunCtx* sctx, RAI_Error* err);
RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script);

#endif /* SRC_SCRIPT_H_ */
