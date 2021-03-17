#pragma once

#include "redis_ai_objects/model.h"

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
