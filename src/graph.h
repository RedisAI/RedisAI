/*
 * graph.h
 *
 *  Created on: 28 Nov 2018
 *      Author: root
 */

#ifndef SRC_GRAPH_H_
#define SRC_GRAPH_H_

#include "config.h"
#include "graph_struct.h"
#include "tensor.h"
#include "redismodule.h"

extern RedisModuleType *RedisAI_GraphType;

int RAI_GraphInit(RedisModuleCtx* ctx);
RAI_Graph* RAI_GraphCreate(RAI_Backend backend, RAI_Device device, const char* graphdef, size_t graphlen);
void RAI_GraphFree(RAI_Graph* graph);

RAI_GraphRunCtx* RAI_GraphRunCtxCreate(RAI_Graph* graph);
int RAI_GraphRunCtxAddInput(RAI_GraphRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor);
int RAI_GraphRunCtxAddOutput(RAI_GraphRunCtx* gctx, const char* outputName);
size_t RAI_GraphRunCtxNumOutputs(RAI_GraphRunCtx* gctx);
RAI_Tensor* RAI_GraphRunCtxOutputTensor(RAI_GraphRunCtx* gctx, size_t index);
void RAI_GraphRunCtxFree(RAI_GraphRunCtx* gctx);

int RAI_GraphRun(RAI_GraphRunCtx* gctx);
RAI_Graph* RAI_GraphGetShallowCopy(RAI_Graph* graph);

#endif /* SRC_GRAPH_H_ */
