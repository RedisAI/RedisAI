#ifndef REDISAI_MODELRUN_CTX_H
#define REDISAI_MODELRUN_CTX_H

#include "model.h"

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
void RAI_ModelRunCtxFree(RAI_ModelRunCtx *mctx, int freeTensors);

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
 * Helper method to parse AI.MODELRUN arguments
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param mctx Destination Model context to store the parsed data
 * @param outkeys array to store the parsed output keys
 * @param mto model to run the session from
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
int RedisAI_Parse_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                        RAI_ModelRunCtx **mctx, RedisModuleString ***inkeys,
                                        RedisModuleString ***outkeys, RAI_Model **mto,
                                        RAI_Error *error);

/**
 * Extract the params for the ModelCtxRun object from AI.MODELRUN arguments.
 *
 * @param ctx Context in which Redis modules operate
 * @param inkeys Model input tensors keys, as an array of strings
 * @param outkeys Model output tensors keys, as an array of strings
 * @param mctx Destination Model context to store the parsed data
 * @param timeout Indicates weather a timeout argument was given in the command
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise
 */

int ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                          RedisModuleString **outkeys, RAI_ModelRunCtx *mctx);

#endif // REDISAI_MODELRUN_CTX_H
