#include "script.h"
#include "script_struct.h"

#ifdef RAI_TORCH_BACKEND
#include "backends/torch.h"
#endif /* RAI_TORCH_BACKEND */

#include "utils/alloc.h"
#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisAI_ScriptType = NULL;

static void* RAI_Script_RdbLoad(struct RedisModuleIO *io, int encver) {
  if (encver != RAI_ENC_VER) {
      /* We should actually log an error here, or try to implement
         the ability to load older versions of our data structure. */
      return NULL;
  }

  RAI_Error err = {0};

  RAI_Device device = RedisModule_LoadUnsigned(io);
  size_t len;
  char *scriptdef = RedisModule_LoadStringBuffer(io, &len);

  RAI_Script *script = RAI_ScriptCreate(device, scriptdef, &err);

  RedisModule_Free(scriptdef);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }

  return script;
}

static void RAI_Script_RdbSave(RedisModuleIO *io, void *value) {
  RAI_Script *script = (RAI_Script*)value;

  size_t len = strlen(script->scriptdef);

  RedisModule_SaveUnsigned(io, script->device);
  RedisModule_SaveStringBuffer(io, script->scriptdef, len);
}

static void RAI_Script_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Script *script = (RAI_Script*)value;

  RedisModule_EmitAOF(aof, "AI.SCRIPTSET", "slc", key, script->device, script->scriptdef);
}

static void RAI_Script_DTFree(void *value) {
  RAI_Error err = {0};
  RAI_ScriptFree(value, &err);
  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }
}

int RAI_ScriptInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmScript = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RAI_Script_RdbLoad,
      .rdb_save = RAI_Script_RdbSave,
      .aof_rewrite = RAI_Script_AofRewrite,
      .mem_usage = NULL,
      .free = RAI_Script_DTFree,
      .digest = NULL
  };

  RedisAI_ScriptType = RedisModule_CreateDataType(ctx, "AI_SCRIPT", 0, &tmScript);
  return RedisAI_ScriptType != NULL;
}

RAI_Script *RAI_ScriptCreate(RAI_Device device, const char *scriptdef, RAI_Error* err) {

  return RAI_ScriptCreateTorch(device, scriptdef, err);
}

void RAI_ScriptFree(RAI_Script* script, RAI_Error* err) {
  if (--script->refCount > 0){
    return;
  }

  RAI_ScriptFreeTorch(script, err);
}

RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script) {
#define PARAM_INITIAL_SIZE 10
  RAI_ScriptRunCtx* sctx = RedisModule_Calloc(1, sizeof(*sctx));
  sctx->script = RAI_ScriptGetShallowCopy(script);
  sctx->inputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
  sctx->outputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
  return sctx;
}

static int Script_RunCtxAddParam(RAI_ScriptRunCtx* sctx, RAI_ScriptCtxParam* paramArr,
                                 RAI_Tensor* tensor) {

  RAI_ScriptCtxParam param = {
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, RAI_Tensor* inputTensor) {
  return Script_RunCtxAddParam(sctx, sctx->inputs, inputTensor);
}

int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx) {
  return Script_RunCtxAddParam(sctx, sctx->outputs, NULL);
}

size_t RAI_ScriptRunCtxNumOutputs(RAI_ScriptRunCtx* sctx) {
  return array_len(sctx->outputs);
}

RAI_Tensor* RAI_ScriptRunCtxOutputTensor(RAI_ScriptRunCtx* sctx, size_t index) {
  assert(RAI_ScriptRunCtxNumOutputs(sctx) > index && index >= 0);
  return sctx->outputs[index].tensor;
}

void RAI_ScriptRunCtxFree(RAI_ScriptRunCtx* sctx) {
  for (size_t i = 0 ; i < array_len(sctx->inputs) ; ++i) {
    RAI_TensorFree(sctx->inputs[i].tensor);
  }
  array_free(sctx->inputs);

  for (size_t i = 0 ; i < array_len(sctx->outputs) ; ++i) {
    if (sctx->outputs[i].tensor) {
      RAI_TensorFree(sctx->outputs[i].tensor);
    }
  }
  array_free(sctx->outputs);

  RAI_Error err = {0};
  RAI_ScriptFree(sctx->script, &err);

  if (err.code != RAI_OK) {
    // TODO: take it to client somehow
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }
}

int RAI_ScriptRun(RAI_ScriptRunCtx* sctx, RAI_Error* err) {
  return RAI_ScriptRunTorch(sctx, err);
}

RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script) {
  ++script->refCount;
  return script;
}
