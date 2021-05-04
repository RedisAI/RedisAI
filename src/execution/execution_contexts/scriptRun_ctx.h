#pragma once
#include "redis_ai_objects/script.h"

/**
 * Allocates the RAI_ScriptRunCtx data structure required for async background
 * work within `RedisAI_RunInfo` structure on RedisAI blocking commands
 *
 * @param script input script
 * @param fnname function name to used from the script
 * @return RAI_ScriptRunCtx to be used within
 */
RAI_ScriptRunCtx *RAI_ScriptRunCtxCreate(RAI_Script *script, const char *fnname);

/**
 * Allocates a RAI_ScriptCtxParam data structure, and enforces a shallow copy of
 * the provided input tensor, adding it to the input tensors array of the
 * RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the input tensor
 * @param inputTensor input tensor structure
 * @return returns 1 on success, 0 in case of error.
 */
int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor, RAI_Error *error);

/**
 * For each Allocates a RAI_ScriptCtxParam data structure, and enforces a
 * shallow copy of the provided input tensor, adding it to the input tensors
 * array of the RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the input tensor
 * @param inputTensors input tensors array
 * @param len input tensors array len
 * @return returns 1 on success, 0 in case of error.
 */
int RAI_ScriptRunCtxAddInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors, size_t len,
                                 RAI_Error *error);

/**
 * @brief Adds a list length to the given script contxt.
 *
 * @param sctx input RAI_ScriptRunCtx to add the list len.
 * @param len input tensors array len
 * @return int returns 1 on success, 0 in case of error.
 */
int RAI_ScriptRunCtxAddListSize(RAI_ScriptRunCtx *sctx, size_t len);

/**
 * Allocates a RAI_ScriptCtxParam data structure, and sets the tensor reference
 * to NULL ( will be set after SCRIPTRUN ), adding it to the outputs tensors
 * array of the RAI_ScriptRunCtx.
 *
 * @param sctx input RAI_ScriptRunCtx to add the output tensor
 * @return returns 1 on success ( always returns success )
 */
int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx *sctx);

/**
 * Returns the total number of output tensors of the RAI_ScriptRunCtx
 *
 * @param sctx RAI_ScriptRunCtx
 * @return the total number of output tensors of the RAI_ScriptRunCtx
 */
size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx *sctx);

/**
 * Get the RAI_Tensor at the output array index position
 *
 * @param sctx RAI_ScriptRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor *RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx *sctx, size_t index);

/**
 * Frees the RAI_ScriptRunCtx data structure used within for async background
 * work
 *
 * @param sctx
 */
void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx *sctx);

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
int RAI_ScriptRun(RAI_ScriptRunCtx *sctx, RAI_Error *err);

/**
 * Insert the ScriptRunCtx to the run queues so it will run asynchronously.
 *
 * @param sctx ScriptRunCtx to execute
 * @param ScriptAsyncFinish A callback that will be called when the execution is finished.
 * @param private_data This is going to be sent to to the ScriptAsyncFinish.
 * @return REDISMODULE_OK if the sctx was insert to the queues successfully, REDISMODULE_ERR
 * otherwise.
 */
int RAI_ScriptRunAsync(RAI_ScriptRunCtx *sctx, RAI_OnFinishCB ScriptAsyncFinish,
                       void *private_data);

/**
 * @brief Retuens the current Script Run context function signature
 *
 * @param sctx ScriptRunCtx
 * @return TorchScriptFunctionArgumentType* Null in case of no match, arr of argument type according
 * to function signature
 */
TorchScriptFunctionArgumentType *RAI_ScriptRunCtxGetSignature(RAI_ScriptRunCtx *sctx);

/**
 * @brief Returns the length of the input list in the given index.
 *
 * @param sctx ScriptRunCtx.
 * @param index Index of the list out of all the lists given as inputs.
 * @return size_t length of the input list in the given index.
 */
size_t RAI_ScriptRunCtxGetInputListLen(RAI_ScriptRunCtx *sctx, size_t index);

/**
 * Extract the ternsor parameters for the ScriptCtxRun object from AI.SCRIPTEXECUTE arguments.
 *
 * @param ctx Context in which Redis modules operate.
 * @param inkeys Script input tensors keys, as an array of strings.
 * @param outkeys Script output tensors keys, as an array of strings.
 * @param sctx Destination Script context to store the parsed data.
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise.
 */

int ScriptRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                           RedisModuleString **outkeys, RAI_ScriptRunCtx *sctx, RAI_Error *err);

int RAI_ScriptRunCtxAddTensorInput(RAI_ScriptRunCtx *sctx, RAI_Tensor *inputTensor);

int RAI_ScriptRunCtxAddIntInput(RAI_ScriptRunCtx *sctx, int32_t i);

int RAI_ScriptRunCtxAddFloatInput(RAI_ScriptRunCtx *sctx, float f);

int RAI_ScriptRunCtxAddRStringInput(RAI_ScriptRunCtx *sctx, RedisModuleString *s);

int RAI_ScriptRunCtxAddStringInput(RAI_ScriptRunCtx *sctx, const char *s, size_t len);

int RAI_ScriptRunCtxAddTensorInputList(RAI_ScriptRunCtx *sctx, RAI_Tensor **inputTensors,
                                       size_t count);

int RAI_ScriptRunCtxAddIntInputList(RAI_ScriptRunCtx *sctx, int32_t *intInputs, size_t count);

int RAI_ScriptRunCtxAddFloatInputList(RAI_ScriptRunCtx *sctx, float *floatInputs, size_t count);

int RAI_ScriptRunCtxAddRStringInputList(RAI_ScriptRunCtx *sctx, RedisModuleString **stringInputs,
                                        size_t count);

int RAI_ScriptRunCtxAddStringInputList(RAI_ScriptRunCtx *sctx, const char **stringInputs,
                                       size_t *lens, size_t count);
                                       