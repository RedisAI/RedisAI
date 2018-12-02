#include "graph.h"
#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisDL_GraphType = NULL;

typedef struct RDL_Graph{
  TF_Graph* graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  size_t refCount;
}RDL_Graph;

typedef struct RDL_GraphCtxParam{
  TF_Output name;
  RDL_Tensor* tensor;
}RDL_GraphCtxParam;

typedef struct RDL_GraphRunCtx{
  RDL_Graph* graph;
  RDL_GraphCtxParam* inputs;
  RDL_GraphCtxParam* outputs;
}RDL_GraphRunCtx;

static void* Graph_RdbLoad(struct RedisModuleIO *io, int encver){
  //todo
  return NULL;
}

static void Graph_RdbSave(RedisModuleIO *rdb, void *value){
  //todo
}

static void Graph_DTFree(void *value){
  RDL_GraphFree(value);
}

int RDL_GraphInit(RedisModuleCtx* ctx){
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

RDL_Graph* RDL_GraphCreate(const char* prefix, const char* graphdef, size_t graphlen){
  TF_Graph* graph = TF_NewGraph();

  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(options, prefix);

  TF_Buffer *buffer = TF_NewBuffer();
  buffer->length = graphlen;
  buffer->data = graphdef;

  TF_Status *status = TF_NewStatus();

  TF_GraphImportGraphDef(graph, buffer, options, status);

  if (TF_GetCode(status) != TF_OK) {
    // todo: free memory
    return NULL;
  }

  TF_DeleteImportGraphDefOptions(options);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(status);

  TF_Status *sessionStatus = TF_NewStatus();

  TF_SessionOptions *sessionOptions = TF_NewSessionOptions();
  TF_Session *session = TF_NewSession(graph, sessionOptions, sessionStatus);

  if (TF_GetCode(sessionStatus) != TF_OK) {
    // TODO: free memory
    return NULL;
  }

  TF_DeleteSessionOptions(sessionOptions);
  TF_DeleteStatus(sessionStatus);

  RDL_Graph* ret = RedisModule_Alloc(sizeof(*ret));
  ret->graph = graph;
  ret->session = session;
  ret->refCount = 1;

  return ret;
}

void RDL_GraphFree(RDL_Graph* graph){
  if(--graph->refCount > 0){
    return;
  }
  TF_Status *status = TF_NewStatus();
  TF_CloseSession(graph->session, status);

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteSession(graph->session, status);
  graph->session = NULL;

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteGraph(graph->graph);
  graph->graph = NULL;

  TF_DeleteStatus(status);

  RedisModule_Free(graph);
}

RDL_GraphRunCtx* RDL_RunCtxCreate(RDL_Graph* graph){
#define PARAM_INITIAL_SIZE 10
  RDL_GraphRunCtx* gctx = RedisModule_Alloc(sizeof(*gctx));
  gctx->graph = RDL_GraphGetShallowCopy(graph);
  gctx->inputs = array_new(RDL_GraphCtxParam, PARAM_INITIAL_SIZE);
  gctx->outputs = array_new(RDL_GraphCtxParam, PARAM_INITIAL_SIZE);
  return gctx;
}

static int Graph_RunCtxAddParam(RDL_GraphRunCtx* gctx, RDL_GraphCtxParam* paramArr, const char* name, RDL_Tensor* tensor){
  TF_Output port;
  port.oper = TF_GraphOperationByName(gctx->graph->graph, name);
  port.index = 0;
  if(port.oper == NULL){
    return 0;
  }
  RDL_GraphCtxParam param = {
      .name = port,
      .tensor = tensor ? RDL_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RDL_RunCtxAddInput(RDL_GraphRunCtx* gctx, const char* inputName, RDL_Tensor* inputTensor){
  return Graph_RunCtxAddParam(gctx, gctx->inputs, inputName, inputTensor);
}

int RDL_RunCtxAddOutput(RDL_GraphRunCtx* gctx, const char* outputName){
  return Graph_RunCtxAddParam(gctx, gctx->outputs, outputName, NULL);
}

size_t RDL_RunCtxNumOutputs(RDL_GraphRunCtx* gctx){
  return array_len(gctx->outputs);
}

RDL_Tensor* RDL_RunCtxOutputTensor(RDL_GraphRunCtx* gctx, size_t index){
  assert(RDL_RunCtxNumOutputs(gctx) > index && index >= 0);
  return gctx->outputs[index].tensor;
}

void RDL_RunCtxFree(RDL_GraphRunCtx* gctx){
  for(size_t i = 0 ; i < array_len(gctx->inputs) ; ++i){
    RDL_TensorFree(gctx->inputs[i].tensor);
  }
  array_free(gctx->inputs);

  for(size_t i = 0 ; i < array_len(gctx->outputs) ; ++i){
    if(gctx->outputs[i].tensor){
      RDL_TensorFree(gctx->outputs[i].tensor);
    }
  }
  array_free(gctx->outputs);

  RDL_GraphFree(gctx->graph);
}

int RDL_GraphRun(RDL_GraphRunCtx* gctx){
  TF_Status *status = TF_NewStatus();

  TF_Tensor* inputTensorsValues[array_len(gctx->inputs)];
  TF_Output inputs[array_len(gctx->inputs)];
  TF_Tensor* outputTensorsValues[array_len(gctx->outputs)];
  TF_Output outputs[array_len(gctx->outputs)];

  for(size_t i = 0 ; i < array_len(gctx->inputs) ; ++i){
    inputTensorsValues[i] = RDL_TensorGetTensor(gctx->inputs[i].tensor);
    inputs[i] = gctx->inputs[i].name;
  }

  for(size_t i = 0 ; i < array_len(gctx->outputs) ; ++i){
    outputs[i] = gctx->outputs[i].name;
  }

  TF_SessionRun(gctx->graph->session, NULL /* run_options */,
                inputs, inputTensorsValues, array_len(gctx->inputs),
                outputs, outputTensorsValues, array_len(gctx->outputs),
                NULL /* target_opers */, 0 /* ntargets */,
                NULL /* run_Metadata */,
                status);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteStatus(status);
    return 0;
  }

  for(size_t i = 0 ; i < array_len(gctx->outputs) ; ++i){
    gctx->outputs[i].tensor = RDL_TensorCreateFromTensor(outputTensorsValues[i]);
  }

  TF_DeleteStatus(status);
  return 1;
}

RDL_Graph* RDL_GraphGetShallowCopy(RDL_Graph* graph){
  ++graph->refCount;
  return graph;
}
