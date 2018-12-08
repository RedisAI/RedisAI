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

extern RedisModuleType *RedisDL_GraphType;

int RDL_GraphInit(RedisModuleCtx* ctx);
RDL_Graph* RDL_GraphCreate(const char* prefix, RDL_Backend backend, const char* graphdef, size_t graphlen);
void RDL_GraphFree(RDL_Graph* graph);
RDL_GraphRunCtx* RDL_RunCtxCreate(RDL_Graph* graph);
int RDL_RunCtxAddInput(RDL_GraphRunCtx* gctx, const char* inputName, RDL_Tensor* inputTensor);
int RDL_RunCtxAddOutput(RDL_GraphRunCtx* gctx, const char* outputName);
size_t RDL_RunCtxNumOutputs(RDL_GraphRunCtx* gctx);
RDL_Tensor* RDL_RunCtxOutputTensor(RDL_GraphRunCtx* gctx, size_t index);
void RDL_RunCtxFree(RDL_GraphRunCtx* gctx);
int RDL_GraphRun(RDL_GraphRunCtx* gctx);
RDL_Graph* RDL_GraphGetShallowCopy(RDL_Graph* graph);

#endif /* SRC_GRAPH_H_ */
