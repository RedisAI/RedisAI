#include "model.h"
#include "model_struct.h"

#ifdef RAI_TENSORFLOW_BACKEND
#include "backends/tensorflow.h"
#endif /* RAI_TENSORFLOW_BACKEND */

#ifdef RAI_TORCH_BACKEND
#include "backends/torch.h"
#endif /* RAI_TORCH_BACKEND */

#include "utils/alloc.h"
#include "utils/arr_rm_alloc.h"

RedisModuleType *RedisAI_ModelType = NULL;

static void* Model_RdbLoad(struct RedisModuleIO *io, int encver) {
  //todo
  return NULL;
}

static void Model_RdbSave(RedisModuleIO *rdb, void *value) {
  //todo
}

static void Model_DTFree(void *value) {
  RAI_ModelFree(value);
}

int RAI_ModelInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmModel = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = Model_RdbLoad,
      .rdb_save = Model_RdbSave,
      .aof_rewrite = NULL,
      .mem_usage = NULL,
      .free = Model_DTFree,
      .digest = NULL
  };

  RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", 0, &tmModel);
  return RedisAI_ModelType != NULL;
}

RAI_Model *RAI_ModelCreate(RAI_Backend backend, RAI_Device device,
                           const char *modeldef, size_t modellen) {
  if (backend == RAI_BACKEND_TENSORFLOW) {
    return RAI_ModelCreateTF(backend, device, modeldef, modellen);
  }
  else if (backend == RAI_BACKEND_TORCH) {
    return RAI_ModelCreateTorch(backend, device, modeldef, modellen);
  }

  printf("ERR: Unsupported backend.\n");
  assert(0);

  return NULL;
}

void RAI_ModelFree(RAI_Model* model) {
  if (--model->refCount > 0){
    return;
  }

  if (model->backend == RAI_BACKEND_TENSORFLOW) {
    RAI_ModelFreeTF(model);
  }
  else if (model->backend == RAI_BACKEND_TORCH) {
    RAI_ModelFreeTorch(model);
  }
  else {
    printf("ERR: Unsupported backend.\n");
    assert(0);
  }

  RedisModule_Free(model);
}

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model) {
#define PARAM_INITIAL_SIZE 10
  RAI_ModelRunCtx* gctx = RedisModule_Alloc(sizeof(*gctx));
  gctx->model = RAI_ModelGetShallowCopy(model);
  gctx->inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  gctx->outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  return gctx;
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx* gctx, RAI_ModelCtxParam* paramArr,
                                const char* name, RAI_Tensor* tensor) {

  RAI_ModelCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  paramArr = array_append(paramArr, param);
  return 1;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* gctx, const char* inputName, RAI_Tensor* inputTensor) {
  return Model_RunCtxAddParam(gctx, gctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* gctx, const char* outputName) {
  return Model_RunCtxAddParam(gctx, gctx->outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* gctx) {
  return array_len(gctx->outputs);
}

RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* gctx, size_t index) {
  assert(RAI_ModelRunCtxNumOutputs(gctx) > index && index >= 0);
  return gctx->outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx* gctx) {
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

  RAI_ModelFree(gctx->model);
}

int RAI_ModelRun(RAI_ModelRunCtx* gctx) {
  int ret;

  switch (gctx->model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      ret = RAI_ModelRunTF(gctx);
      break;
    case RAI_BACKEND_TORCH:
      ret = RAI_ModelRunTorch(gctx);
      break;
    default:
      printf("ERR: Unsupported backend.\n");
      assert(0);
      break;
  }

  return ret;
}

RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model) {
  ++model->refCount;
  return model;
}
