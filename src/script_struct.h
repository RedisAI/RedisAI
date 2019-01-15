#ifndef SRC_SCRIPT_STRUCT_H_
#define SRC_SCRIPT_STRUCT_H_

#include "config.h"
#include "tensor_struct.h"

typedef struct RAI_Script {
  void* script;
  char* scriptdef;
  // TODO: scripts do not have placement in PyTorch
  // Placement depends on the inputs, as do outputs
  // We keep it here at the moment, until we have a
  // CUDA allocator for dlpack
  RAI_Device device;
  long long refCount;
} RAI_Script;

typedef struct RAI_ScriptCtxParam {
  const char* name;
  RAI_Tensor* tensor;
} RAI_ScriptCtxParam;

typedef struct RAI_ScriptRunCtx {
  RAI_Script* script;
  const char* fnname;
  RAI_ScriptCtxParam* inputs;
  RAI_ScriptCtxParam* outputs;
} RAI_ScriptRunCtx;

#endif /* SRC_SCRIPT_STRUCT_H_ */
