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

typedef struct RDL_Graph{
  TF_Graph* graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  size_t refCount;
}RDL_Graph;

extern RedisModuleType *RedisDL_GraphType;

int Graph_Init(RedisModuleCtx* ctx);
RDL_Graph* Graph_Create(const char* prefix, const char* graphdef, size_t graphlen);
void Graph_Free(RDL_Graph* graph);



#endif /* SRC_GRAPH_H_ */
