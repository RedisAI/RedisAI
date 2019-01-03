#ifndef SRC_GRAPH_STRUCT_H_
#define SRC_GRAPH_STRUCT_H_

#include "config.h"
#include "tensor_struct.h"

typedef struct RAI_Graph {
  void* graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  RAI_Backend backend;
  long long refCount;
} RAI_Graph;

typedef struct RAI_GraphCtxParam {
  const char* name;
  RAI_Tensor* tensor;
} RAI_GraphCtxParam;

typedef struct RAI_GraphRunCtx {
  RAI_Graph* graph;
  RAI_GraphCtxParam* inputs;
  RAI_GraphCtxParam* outputs;
} RAI_GraphRunCtx;

#endif /* SRC_GRAPH_STRUCT_H_ */
