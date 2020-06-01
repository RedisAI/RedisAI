/**
 * script.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the PyTorch Script data structure.
 *
 */

#ifndef SRC_SCRIPT_H_
#define SRC_SCRIPT_H_

#include "config.h"
#include "err.h"
#include "redismodule.h"
#include "script_struct.h"
#include "tensor.h"

extern RedisModuleType* RedisAI_ScriptType;

/**
 * Helper method to register the script type exported by the module.
 *
 * @param ctx Context in which Redis modules operate
 * @return
 */
int RAI_ScriptInit(RedisModuleCtx* ctx);

/**
 * Helper method to allocated and initialize a RAI_Script. Relies on Pytorch
 * backend `script_create` callback function.
 *
 * @param devicestr device string
 * @param tag script model tag
 * @param scriptdef encoded script definition
 * @param error error data structure to store error message in the case of
 * failures
 * @return RAI_Script script structure on success, or NULL if failed
 */
RAI_Script* RAI_ScriptCreate(const char* devicestr, const char* tag,
                             const char* scriptdef, RAI_Error* err);

/**
 * Frees the memory of the RAI_Script when the script reference count reaches
 * 0. It is safe to call this function with a NULL input script.
 *
 * @param script input script to be freed
 * @param error error data structure to store error message in the case of
 * failures
 */
void RAI_ScriptFree(RAI_Script* script, RAI_Error* err);

/**
 * Allocates the RAI_ScriptRunCtx data structure required for async background
 * work within `RedisAI_RunInfo` structure on RedisAI blocking commands
 *
 * @param script input script
 * @param fnname function name to used from the script
 * @return RAI_ScriptRunCtx to be used within
 */
RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script,
                                         const char* fnname);

/**
 * Allocates a RAI_ScriptCtxParam data structure, and enforces a shallow copy of
 * the provided input tensor, adding it to the input tensors array of the
 * RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the input tensor
 * @param inputTensor input tensor structure
 * @param err error data structure to store error message in the case of
 * failures
 * @return returns 1 on success, 0 in case of error.
 */
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, RAI_Tensor* inputTensor, RAI_Error* err);

/**
 * For each Allocates a RAI_ScriptCtxParam data structure, and enforces a shallow copy of
 * the provided input tensor, adding it to the input tensors array of the
 * RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the input tensor
 * @param inputTensors input tensors array
 * @param len input tensors array len
 * @param err error data structure to store error message in the case of
 * failures
 * @return returns 1 on success, 0 in case of error.
 */
int RAI_ScriptRunCtxAddInputList(RAI_ScriptRunCtx* sctx, RAI_Tensor** inputTensors, size_t len, RAI_Error* err);

/**
 * Allocates a RAI_ScriptCtxParam data structure, and sets the tensor reference
 * to NULL ( will be set after SCRIPTRUN ), adding it to the outputs tensors
 * array of the RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the output tensor
 * @return returns 1 on success ( always returns success )
 */
int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx);

/**
 * Returns the total number of output tensors of the RAI_ScriptRunCtx
 *
 * @param sctx RAI_ScriptRunCtx
 * @return the total number of output tensors of the RAI_ScriptRunCtx
 */
size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx* sctx);

/**
 * Get the RAI_Tensor at the output array index position
 *
 * @param sctx RAI_ScriptRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor* RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx* sctx, size_t index);

/**
 * Frees the RAI_ScriptRunCtx data structure used within for async background
 * work
 *
 * @param sctx
 * @param freeTensors free input and output tensors or leave them allocated
 */
void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx* sctx, int freeTensors);

/**
 * Given the input script context, run associated script
 * session. On success, the tensors corresponding to outputs[0,noutputs-1] are
 * placed in the RAI_ScriptRunCtx output tensors array. Relies on PyTorch's
 * `script_run` definition.
 *
 * @param sctx input script context
 * @param error error data structure to store error message in the case of
 * failures
 * @return REDISMODULE_OK if the underlying backend `script_run` ran
 * successfully, or REDISMODULE_ERR if failed.
 */
int RAI_ScriptRun(RAI_ScriptRunCtx* sctx, RAI_Error* err);

/**
 * Every call to this function, will make the RAI_Script 'script' requiring an
 * additional call to RAI_ScriptFree() in order to really free the script.
 * Returns a shallow copy of the script.
 *
 * @param script input script
 * @return script
 */
RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script);

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */

/**
 * Helper method to get a Script from keyspace. In the case of failure the key
 * is closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key script's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param script destination script structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the script value stored at key was correctly
 * returned and available at *script variable, or REDISMODULE_ERR if there was
 * an error getting the Script
 */
int RAI_GetScriptFromKeyspace(RedisModuleCtx* ctx, RedisModuleString* keyName,
                              RedisModuleKey** key, RAI_Script** script,
                              int mode);


/**
 * Helper method to parse AI.SCRIPTRUN arguments
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param sctx Destination Script context to store the parsed data
 * @param outkeys array to store the parsed output keys
 * @param sto script to run the session from
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RedisAI_Parse_ScriptRun_RedisCommand(RedisModuleCtx *ctx,
                                         RedisModuleString **argv, int argc,
                                         RAI_ScriptRunCtx **sctx,
                                         RedisModuleString ***inkeys,
                                         RedisModuleString ***outkeys,
                                         struct RAI_Script **sto,
                                         RAI_Error *error);


#if 0
/**
 * Helper method to reply if the ctx is not NULL or fallback and set the error in the RAI_Error structure
 * @param ctx Context in which Redis modules operate
 * @param error the RAI_Error data structure to be populated with the error details in case ctx is NULL
 * @param code the error code
 * @param errorMessage the error detail
 */
void RedisAI_ReplyOrSetError(RedisModuleCtx *ctx, RAI_Error *error, RAI_ErrorCode code, const char* errorMessage );
#endif

/**
 * @brief  Returns the redis module type representing a script.
 * @return redis module type representing a script.
 */
RedisModuleType *RAI_ScriptRedisType(void);

#endif /* SRC_SCRIPT_H_ */
