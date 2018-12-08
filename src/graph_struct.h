#ifndef SRC_GRAPH_STRUCT_H_
#define SRC_GRAPH_STRUCT_H_

#include "config.h"
#include "tensor_struct.h"

typedef struct RDL_Graph {
  void* graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  RDL_Backend backend;
  long long refCount;
} RDL_Graph;

typedef struct RDL_GraphCtxParam {
  const char* name;
  RDL_Tensor* tensor;
} RDL_GraphCtxParam;

typedef struct RDL_GraphRunCtx {
  RDL_Graph* graph;
  RDL_GraphCtxParam* inputs;
  RDL_GraphCtxParam* outputs;
} RDL_GraphRunCtx;

#endif /* SRC_GRAPH_STRUCT_H_ */