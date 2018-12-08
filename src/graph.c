#include "graph.h"
#include "graph_struct.h"

#ifdef RDL_TENSORFLOW_BACKEND
#include "backends_tensorflow.h"
#endif /* RDL_TENSORFLOW_BACKEND */

#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisDL_GraphType = NULL;

static void* Graph_RdbLoad(struct RedisModuleIO *io, int encver) {
  //todo
  return NULL;
}

static void Graph_RdbSave(RedisModuleIO *rdb, void *value) {
  //todo
}

static void Graph_DTFree(void *value) {
  RDL_GraphFree(value);
}

int RDL_GraphInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmGraph = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Graph_RdbLoad,
      .rdb_save = Graph_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Graph_DTFree,
      .digest = NULL
  };

  RedisDL_GraphType = RedisModule_CreateDataType(ctx, "DL__GRAPH", 0, &tmGraph);
  return RedisDL_GraphType != NULL;
}

RDL_Graph *RDL_GraphCreate(const char *prefix, RDL_Backend backend,
                           const char *graphdef, size_t graphlen) {
  if (backend == RDL_BACKEND_TENSORFLOW) {
    return RDL_GraphCreateTF(prefix, backend, graphdef, graphlen);
  }

  return NULL;
}

void RDL_GraphFree(RDL_Graph* graph) {
  if (--graph->refCount > 0){
    return;
  }

  if (graph->backend == RDL_BACKEND_TENSORFLOW) {
    RDL_GraphFreeTF(graph);
  }
  else {
    // TODO: err properly
    printf("ERR: Unsupported backend.\n");
  }

  RedisModule_Free(graph);
}

RDL_GraphRunCtx* RDL_RunCtxCreate(RDL_Graph* graph) {
#define PARAM_INITIAL_SIZE 10
  RDL_GraphRunCtx* gctx = RedisModule_Alloc(sizeof(*gctx));
  gctx->graph = RDL_GraphGetShallowCopy(graph);
  gctx->inputs = array_new(RDL_GraphCtxParam, PARAM_INITIAL_SIZE);
  gctx->outputs = array_new(RDL_GraphCtxParam, PARAM_INITIAL_SIZE);
  return gctx;
}

static int Graph_RunCtxAddParam(RDL_GraphRunCtx* gctx, RDL_GraphCtxParam* paramArr,
                                const char* name, RDL_Tensor* tensor) {

  RDL_GraphCtxParam param = {
      .name = name,
      .tensor = tensor ? RDL_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RDL_RunCtxAddInput(RDL_GraphRunCtx* gctx, const char* inputName, RDL_Tensor* inputTensor) {
  return Graph_RunCtxAddParam(gctx, gctx->inputs, inputName, inputTensor);
}

int RDL_RunCtxAddOutput(RDL_GraphRunCtx* gctx, const char* outputName) {
  return Graph_RunCtxAddParam(gctx, gctx->outputs, outputName, NULL);
}

size_t RDL_RunCtxNumOutputs(RDL_GraphRunCtx* gctx) {
  return array_len(gctx->outputs);
}

RDL_Tensor* RDL_RunCtxOutputTensor(RDL_GraphRunCtx* gctx, size_t index) {
  assert(RDL_RunCtxNumOutputs(gctx) > index && index >= 0);
  return gctx->outputs[index].tensor;
}

void RDL_RunCtxFree(RDL_GraphRunCtx* gctx) {
  for (size_t i = 0 ; i < array_len(gctx->inputs) ; ++i) {
    RDL_TensorFree(gctx->inputs[i].tensor);
  }
  array_free(gctx->inputs);

  for (size_t i = 0 ; i < array_len(gctx->outputs) ; ++i) {
    if (gctx->outputs[i].tensor) {
      RDL_TensorFree(gctx->outputs[i].tensor);
    }
  }
  array_free(gctx->outputs);

  RDL_GraphFree(gctx->graph);
}

int RDL_GraphRun(RDL_GraphRunCtx* gctx) {
  int ret;

  if (gctx->graph->backend == RDL_BACKEND_TENSORFLOW) {
    ret = RDL_GraphRunTF(gctx);
  }

  return ret;
}

RDL_Graph* RDL_GraphGetShallowCopy(RDL_Graph* graph) {
  ++graph->refCount;
  return graph;
}
