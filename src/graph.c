#include "graph.h"
#include "graph_struct.h"

#ifdef RAI_TENSORFLOW_BACKEND
#include "backends/tensorflow.h"
#endif /* RAI_TENSORFLOW_BACKEND */

#ifdef RAI_TORCH_BACKEND
#include "backends/torch.h"
#endif /* RAI_TORCH_BACKEND */

#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisAI_GraphType = NULL;

static void* Graph_RdbLoad(struct RedisModuleIO *io, int encver) {
  //todo
  return NULL;
}

static void Graph_RdbSave(RedisModuleIO *rdb, void *value) {
  //todo
}

static void Graph_DTFree(void *value) {
  RAI_GraphFree(value);
}

int RAI_GraphInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmGraph = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Graph_RdbLoad,
      .rdb_save = Graph_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Graph_DTFree,
      .digest = NULL
  };

  RedisAI_GraphType = RedisModule_CreateDataType(ctx, "AI__GRAPH", 0, &tmGraph);
  return RedisAI_GraphType != NULL;
}

RAI_Graph *RAI_GraphCreate(RAI_Backend backend, RAI_Device device,
                           const char *graphdef, size_t graphlen) {
  if (backend == RAI_BACKEND_TENSORFLOW) {
    return RAI_GraphCreateTF(backend, device, graphdef, graphlen);
  }

  return NULL;
}

void RAI_GraphFree(RAI_Graph* graph) {
  if (--graph->refCount > 0){
    return;
  }

  if (graph->backend == RAI_BACKEND_TENSORFLOW) {
    RAI_GraphFreeTF(graph);
  }
  else {
    // TODO: err properly
    printf("ERR: Unsupported backend.\n");
  }

  RedisModule_Free(graph);
}

RAI_GraphRunCtx* RAI_GraphRunCtxCreate(RAI_Graph* graph) {
#define PARAM_INITIAL_SIZE 10
  RAI_GraphRunCtx* gctx = RedisModule_Alloc(sizeof(*gctx));
  gctx->graph = RAI_GraphGetShallowCopy(graph);
  gctx->inputs = array_new(RAI_GraphCtxParam, PARAM_INITIAL_SIZE);
  gctx->outputs = array_new(RAI_GraphCtxParam, PARAM_INITIAL_SIZE);
  return gctx;
}

static int Graph_RunCtxAddParam(RAI_GraphRunCtx* gctx, RAI_GraphCtxParam* paramArr,
                                const char* name, RAI_Tensor* tensor) {

  RAI_GraphCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RAI_GraphRunCtxAddInput(RAI_GraphRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor) {
  return Graph_RunCtxAddParam(gctx, gctx->inputs, inputName, inputTensor);
}

int RAI_GraphRunCtxAddOutput(RAI_GraphRunCtx* gctx, const char* outputName) {
  return Graph_RunCtxAddParam(gctx, gctx->outputs, outputName, NULL);
}

size_t RAI_GraphRunCtxNumOutputs(RAI_GraphRunCtx* gctx) {
  return array_len(gctx->outputs);
}

RAI_Tensor* RAI_GraphRunCtxOutputTensor(RAI_GraphRunCtx* gctx, size_t index) {
  assert(RAI_GraphRunCtxNumOutputs(gctx) > index && index >= 0);
  return gctx->outputs[index].tensor;
}

void RAI_GraphRunCtxFree(RAI_GraphRunCtx* gctx) {
  for (size_t i = 0 ; i < array_len(gctx->inputs) ; ++i) {
    RAI_TensorFree(gctx->inputs[i].tensor);
  }
  array_free(gctx->inputs);

  for (size_t i = 0 ; i < array_len(gctx->outputs) ; ++i) {
    if (gctx->outputs[i].tensor) {
      RAI_TensorFree(gctx->outputs[i].tensor);
    }
  }
  array_free(gctx->outputs);

  RAI_GraphFree(gctx->graph);
}

int RAI_GraphRun(RAI_GraphRunCtx* gctx) {
  int ret;

  if (gctx->graph->backend == RAI_BACKEND_TENSORFLOW) {
    ret = RAI_GraphRunTF(gctx);
  }

  return ret;
}

RAI_Graph* RAI_GraphGetShallowCopy(RAI_Graph* graph) {
  ++graph->refCount;
  return graph;
}
