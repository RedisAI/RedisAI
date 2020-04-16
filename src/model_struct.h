#ifndef SRC_MODEL_STRUCT_H_
#define SRC_MODEL_STRUCT_H_

#include "config.h"
#include "tensor_struct.h"

typedef struct RAI_ModelOpts {
  size_t batchsize;
  size_t minbatchsize;
} RAI_ModelOpts;

typedef struct RAI_Model {
  void* model;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
  RAI_Backend backend;
  char* devicestr;
  char* tag;
  RAI_ModelOpts opts;
  char **inputs;
  size_t ninputs;
  char **outputs;
  size_t noutputs;
  long long refCount;
  void* data;
  void* infokey;
} RAI_Model;

typedef struct RAI_ModelCtxParam {
  const char* name;
  RAI_Tensor* tensor;
} RAI_ModelCtxParam;

typedef struct RAI_ModelRunCtx {
  size_t ctxtype;
  RAI_Model* model;
  RAI_ModelCtxParam* inputs;
  RAI_ModelCtxParam* outputs;
} RAI_ModelRunCtx;

#endif /* SRC_MODEL_STRUCT_H_ */
