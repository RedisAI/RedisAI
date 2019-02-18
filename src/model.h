/*
 * model.h
 *
 *  Created on: 28 Nov 2018
 *      Author: root
 */

#ifndef SRC_MODEL_H_
#define SRC_MODEL_H_

#include "config.h"
#include "model_struct.h"
#include "tensor.h"
#include "redismodule.h"

extern RedisModuleType *RedisAI_ModelType;

int RAI_ModelInit(RedisModuleCtx* ctx);
RAI_Model* RAI_ModelCreate(RAI_Backend backend, RAI_Device device, const char* modeldef, size_t modellen);
void RAI_ModelFree(RAI_Model* model);

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model);
int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor);
int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* gctx, const char* outputName);
size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* gctx);
RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* gctx, size_t index);
void RAI_ModelRunCtxFree(RAI_ModelRunCtx* gctx);

int RAI_ModelRun(RAI_ModelRunCtx* gctx);
RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model);

#endif /* SRC_MODEL_H_ */
