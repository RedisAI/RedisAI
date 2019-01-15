#ifndef SRC_SCRIPT_H_
#define SRC_SCRIPT_H_

#include "config.h"
#include "script_struct.h"
#include "tensor.h"
#include "redismodule.h"

extern RedisModuleType *RedisAI_ScriptType;

int RAI_ScriptInit(RedisModuleCtx* ctx);
RAI_Script* RAI_ScriptCreate(RAI_Device device, const char* scriptdef);
void RAI_ScriptFree(RAI_Script* script);

RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script);
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, const char* inputName, RAI_Tensor* inputTensor);
int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx, const char* outputName);
size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx* sctx);
RAI_Tensor* RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx* sctx, size_t index);
void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx* sctx);

int RAI_ScriptRun(RAI_ScriptRunCtx* sctx);
RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script);

#endif /* SRC_SCRIPT_H_ */
