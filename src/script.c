#include "script.h"
#include "script_struct.h"

#ifdef RAI_TORCH_BACKEND
#include "backends/torch.h"
#endif /* RAI_TORCH_BACKEND */

#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisAI_ScriptType = NULL;

static void* Script_RdbLoad(struct RedisModuleIO *io, int encver) {
  //todo
  return NULL;
}

static void Script_RdbSave(RedisModuleIO *rdb, void *value) {
  //todo
}

static void Script_DTFree(void *value) {
  RAI_ScriptFree(value);
}

int RAI_ScriptInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmScript = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Script_RdbLoad,
      .rdb_save = Script_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Script_DTFree,
      .digest = NULL
  };

  RedisAI_ScriptType = RedisModule_CreateDataType(ctx, "AI_SCRIPT", 0, &tmScript);
  return RedisAI_ScriptType != NULL;
}

RAI_Script *RAI_ScriptCreate(RAI_Device device, const char *scriptdef) {

  return RAI_ScriptCreateTorch(device, scriptdef);
}

void RAI_ScriptFree(RAI_Script* script) {
  if (--script->refCount > 0){
    return;
  }

  RAI_ScriptFreeTorch(script);
}

RAI_ScriptRunCtx* RAI_ScriptRunCtxCreate(RAI_Script* script) {
#define PARAM_INITIAL_SIZE 10
  RAI_ScriptRunCtx* sctx = RedisModule_Alloc(sizeof(*sctx));
  sctx->script = RAI_ScriptGetShallowCopy(script);
  sctx->inputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
  sctx->outputs = array_new(RAI_ScriptCtxParam, PARAM_INITIAL_SIZE);
  return sctx;
}

static int Script_RunCtxAddParam(RAI_ScriptRunCtx* sctx, RAI_ScriptCtxParam* paramArr,
                                 const char* name, RAI_Tensor* tensor) {

  RAI_ScriptCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RAI_ScriptRunCtxAddInput(RAI_ScriptRunCtx* sctx, const char* inputName, RAI_Tensor* inputTensor) {
  return Script_RunCtxAddParam(sctx, sctx->inputs, inputName, inputTensor);
}

int RAI_ScriptRunCtxAddOutput(RAI_ScriptRunCtx* sctx, const char* outputName) {
  return Script_RunCtxAddParam(sctx, sctx->outputs, outputName, NULL);
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

  RAI_ScriptFree(sctx->script);
}

int RAI_ScriptRun(RAI_ScriptRunCtx* sctx) {
  int ret;

  ret = RAI_ScriptRunTorch(sctx);

  return ret;
}

RAI_Script* RAI_ScriptGetShallowCopy(RAI_Script* script) {
  ++script->refCount;
  return script;
}
