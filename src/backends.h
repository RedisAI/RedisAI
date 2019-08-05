#ifndef SRC_BACKENDS_H_
#define SRC_BACKENDS_H_

#include "config.h"
#include "model_struct.h"
#include "script_struct.h"
#include "tensor.h"
#include "err.h"

typedef struct RAI_LoadedBackend {
  RAI_Model* (*model_create_with_nodes)(RAI_Backend, RAI_Device,
                                        size_t, const char**, size_t, const char**,
                                        const char*, size_t, RAI_Error*);
  RAI_Model* (*model_create)(RAI_Backend, RAI_Device,
                             const char*, size_t, RAI_Error*);
  void (*model_free)(RAI_Model*, RAI_Error*);
  int (*model_run)(RAI_ModelRunCtx*, RAI_Error*);
  int (*model_serialize)(RAI_Model*, char**, size_t*, RAI_Error*);

  RAI_Script* (*script_create)(RAI_Device, const char*, RAI_Error*);
  void (*script_free)(RAI_Script*, RAI_Error*);
  int (*script_run)(RAI_ScriptRunCtx*, RAI_Error*);
} RAI_LoadedBackend;

typedef struct RAI_LoadedBackends {
  RAI_LoadedBackend tf;
  RAI_LoadedBackend torch;
  RAI_LoadedBackend onnx;
} RAI_LoadedBackends;

RAI_LoadedBackends RAI_backends;
char* RAI_BackendsPath;

int RAI_LoadBackend(RedisModuleCtx *ctx, int backend, const char *path);

int RAI_LoadDefaultBackend(RedisModuleCtx *ctx, int backend);

#endif