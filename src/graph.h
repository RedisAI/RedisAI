/*
 * graph.h
 *
 *  Created on: 28 Nov 2018
 *      Author: root
 */

#ifndef SRC_GRAPH_H_
#define SRC_GRAPH_H_

#include "tensorflow/c/c_api.h"
#include "redismodule.h"
#include "tensor.h"

typedef struct RDL_Graph{
  TF_Graph* graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  size_t refCount;
}RDL_Graph;

typedef struct RDL_GraphCtxParam RDL_GraphCtxParam;

typedef struct RDL_GraphRunCtx RDL_GraphRunCtx;

extern RedisModuleType *RedisDL_GraphType;

int Graph_Init(RedisModuleCtx* ctx);
RDL_Graph* Graph_Create(const char* prefix, const char* graphdef, size_t graphlen);
void Graph_Free(RDL_Graph* graph);
RDL_GraphRunCtx* Graph_RunCtxCreate(RDL_Graph* graph);
int Graph_RunCtxAddInput(RDL_GraphRunCtx* gctx, const char* inputName, RDL_Tensor* inputTensor);
int Graph_RunCtxAddOutput(RDL_GraphRunCtx* gctx, const char* outputName);
size_t Graph_RunCtxNumOutputs(RDL_GraphRunCtx* gctx);
RDL_Tensor* Graph_RunCtxOutputTensor(RDL_GraphRunCtx* gctx, size_t index);
void Graph_RunCtxFreeInternals(RDL_GraphRunCtx* gctx);
int Graph_Run(RDL_GraphRunCtx* gctx);
RDL_Graph* Graph_GetShallowCopy(RDL_Graph* graph);



#endif /* SRC_GRAPH_H_ */
