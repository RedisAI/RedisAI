#ifndef SRC_SCRIPT_H_
#define SRC_SCRIPT_H_

#include "config.h"
#include "script_struct.h"
#include "tensor.h"
#include "redismodule.h"
#include "err.h"

extern RedisModuleType *RedisAI_ScriptType;

/**
 *
 *
 * @param ctx
 * @return
 */
int RAI_ScriptInit(RedisModuleCtx* ctx);

/**
 *
 *
 * @param devicestr
 * @param tag
 * @param scriptdef
 * @param err
 * @return
 */
RAI_Script* RAI_ScriptCreate( const char* devicestr, const char* tag, const char* scriptdef, RAI_Error* err);

/**
 *
 *
 * @param script
 * @param err
 */
void RAI_ScriptFree(RAI_Script* script, RAI_Error* err);

/**
 *
 *
 * @param script
 * @param fnname
 * @return
 */
RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script, const char *fnname);

/**
 *
 *
 * @param sctx
 * @param inputTensor
 * @return
 */
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, RAI_Tensor* inputTensor);

/**
 *
 *
 * @param sctx
 * @return
 */
int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx);

/**
 *
 *
 * @param sctx
 * @return
 */
size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx* sctx);

/**
 *
 *
 * @param sctx
 * @param index
 * @return
 */
RAI_Tensor* RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx* sctx, size_t index);

/**
 *
 *
 * @param sctx
 */
void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx* sctx);

/**
 *
 *
 * @param sctx
 * @param err
 * @return
 */
int RAI_ScriptRun(RAI_ScriptRunCtx* sctx, RAI_Error* err);

/**
 *
 *
 * @param script
 * @return
 */
RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script);

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
/**
 *
 *
 * @param ctx
 * @param keyName
 * @param key
 * @param script
 * @param mode
 * @return
 */
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key, RAI_Script **script,
                              int mode);

#endif /* SRC_SCRIPT_H_ */
