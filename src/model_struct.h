#ifndef SRC_MODEL_STRUCT_H_
#define SRC_MODEL_STRUCT_H_

#include "config.h"
#include "tensor_struct.h"

typedef struct RAI_ModelOpts {
  size_t batchsize;
  size_t minbatchsize;
  long long backends_intra_op_parallelism;  //  number of threads used within an
  //  individual op for parallelism.
  long long backends_inter_op_parallelism;  //  number of threads used for parallelism
                                            //  between independent operations.
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
  char* data;
  long long datalen;
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
