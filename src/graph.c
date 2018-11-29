#include "graph.h"

RedisModuleType *RedisDL_GraphType = NULL;

static void* Graph_RdbLoad(struct RedisModuleIO *io, int encver){
  //todo
  return NULL;
}

static void Graph_RdbSave(RedisModuleIO *rdb, void *value){
  //todo
}

static void Graph_DTFree(void *value){
  Graph_Free(value);
}

int Graph_Init(RedisModuleCtx* ctx){
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

RDL_Graph* Graph_Create(const char* prefix, const char* graphdef, size_t graphlen){
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

void Graph_Free(RDL_Graph* graph){
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
