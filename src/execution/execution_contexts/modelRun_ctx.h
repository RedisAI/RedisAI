#pragma once

#include "redis_ai_objects/model.h"
#include "execution_ctx.h"

typedef struct RAI_ModelRunCtx {
    RAI_ExecutionCtx* base;
    size_t ctxtype;
    RAI_Model *model;
} RAI_ModelRunCtx;


/**
 * Allocates the RAI_ModelRunCtx data structure required for async background
 * work within `RedisAI_RunInfo` structure on RedisAI blocking commands
 *
 * @param model input model
 * @return RAI_ModelRunCtx to be used within
 */
RAI_ModelRunCtx *RAI_ModelRunCtxCreate(RAI_Model *model);

/**
 * Frees the RAI_ModelRunCtx data structure used within for async background
 * work
 *
 * @param mctx
 * @param freeTensors free input and output tensors or leave them allocated
 */
void RAI_ModelRunCtxFree(RAI_ModelRunCtx *mctxs);

/**
 * Allocates a RAI_ModelCtxParam data structure, and enforces a shallow copy of
 * the provided input tensor, adding it to the input tensors array of the
 * RAI_ModelRunCtx.
 *
 * @param mctx input RAI_ModelRunCtx to add the input tensor
 * @param inputName input tensor name
 * @param inputTensor input tensor structure
 * @return returns 1 on success ( always returns success )
 */
int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx *mctx, const char *inputName, RAI_Tensor *inputTensor);

/**
 * Allocates a RAI_ModelCtxParam data structure, and sets the tensor reference
 * to NULL ( will be set after MODELRUN ), adding it to the outputs tensors
 * array of the RAI_ModelRunCtx.
 *
 * @param mctx RAI_ModelRunCtx to add the output tensor
 * @param outputName output tensor name
 * @return returns 1 on success ( always returns success )
 */
int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx *mctx, const char *outputName);

/**
 * Returns the total number of input tensors of the RAI_ModelRunCtx
 *
 * @param mctx RAI_ModelRunCtx
 * @return the total number of input tensors of the RAI_ModelRunCtx
 */
size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx *mctx);

/**
 * Returns the total number of output tensors of the RAI_ModelCtxParam
 *
 * @param mctx RAI_ModelRunCtx
 * @return the total number of output tensors of the RAI_ModelCtxParam
 */
size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx *mctx);

/**
 * Get the RAI_Tensor at the input array index position
 *
 * @param mctx RAI_ModelRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor *RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx *mctx, size_t index);

/**
 * Get the RAI_Tensor at the output array index position
 *
 * @param mctx RAI_ModelRunCtx
 * @param index input array index position
 * @return RAI_Tensor
 */
RAI_Tensor *RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx *mctx, size_t index);

/**
 * Extract the params for the ModelCtxRun object from AI.MODELEXECUTE arguments.
 *
 * @param ctx Context in which Redis modules operate
 * @param inkeys Model input tensors keys, as an array of strings
 * @param outkeys Model output tensors keys, as an array of strings
 * @param mctx Destination Model context to store the parsed data
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise
 */

// TODO: Remove this once modelrunctx and scriptrunctx have common base struct.
int ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                          RedisModuleString **outkeys, RAI_ModelRunCtx *mctx, RAI_Error *err);

/**
 * Given a model and  the input array of ectxs, run the associated backend
 * session. If the length of the input context is larger than one, then
 * each backend's `model_run` is responsible for concatenating tensors, and run
 * the model in batches with the size of the input array. On success, the
 * tensors corresponding to outputs[0,noutputs-1] are placed in each
 * RAI_ExecutionCtx output tensors array. Relies on each backend's `model_run`
 * definition.
 *
 * @param mctxs array on input model contexts
 * @param n length of input model contexts array
 * @param error error data structure to store error message in the case of
 * failures
 * @return REDISMODULE_OK if the underlying backend `model_run` runned
 * successfully, or REDISMODULE_ERR if failed.
 */
int RAI_ModelRun(RAI_Model* model, RAI_ExecutionCtx **ectxs, long long n, RAI_Error *err);

/**
 * Insert the ModelRunCtx to the run queues so it will run asynchronously.
 *
 * @param mctx ModelRunCtx to execute
 * @param ModelAsyncFinish A callback that will be called when the execution is finished.
 * @param private_data This is going to be sent to to the ModelAsyncFinish.
 * @return REDISMODULE_OK if the mctx was insert to the queues successfully, REDISMODULE_ERR
 * otherwise.
 */

int RAI_ModelRunAsync(RAI_ModelRunCtx *mctx, RAI_OnFinishCB ModelAsyncFinish, void *private_data);

/**
 * @brief Returns the internal RAI_Model object of RAI_ModelRunCtx.
 */
RAI_Model* RAI_ModelRunCtxGetModel(RAI_ModelRunCtx* mctx);
