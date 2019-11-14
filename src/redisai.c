#include "redismodule.h"
#include "tensor.h"
#include "model.h"
#include "script.h"
#include "backends.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "rmutil/args.h"

#define REDISAI_H_INCLUDE
#include "redisai.h"
#undef REDISAI_H_INCLUDE


typedef struct queueItem {
  struct queueItem *next;
  void *value;
} queueItem;

typedef struct queue {
  queueItem *front;
  queueItem *back;
  void (*free)(void *ptr);
  unsigned long len;
} queue;

queue *queueCreate(void) {
  struct queue *queue;

  if ((queue = RedisModule_Calloc(1, sizeof(*queue))) == NULL)
    return NULL;

  queue->front = queue->back = NULL;
  queue->len = 0;
  queue->free = NULL;
  return queue;
}

void queuePush(queue *queue, void *value) {
  queueItem *item;

  if ((item = RedisModule_Calloc(1, sizeof(*item))) == NULL)
    return;
  item->value = value;
  item->next = NULL;

  if (queue->len == 0) {
    queue->front = queue->back = item;
  } else {
    queue->back->next = item;
    queue->back = item;
  }
  queue->len++;
}

queueItem *queuePop(queue *queue) {
  queueItem *item = queue->front;
  if (item == NULL) {
    return NULL;
  }
  queue->front = item->next;
  if (item == queue->back) {
    queue->back = NULL;
  }
  item->next = NULL;
  queue->len--;
  return item;
}

void queueRelease(queue *queue) {
  unsigned long len;
  queueItem *current;

  len = queue->len;
  while(len--) {
    current = queuePop(queue);
    if (current && queue->free) queue->free(current->value);
    RedisModule_Free(current);
  }
  queue->front = queue->back = NULL;
  queue->len = 0;
}

typedef struct RunQueueInfo {
  pthread_mutex_t run_queue_mutex;
  pthread_cond_t queue_condition_var;
  queue *run_queue;
  pthread_t *threads;
} RunQueueInfo;

static AI_dict *run_queues = NULL;
static int perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;

int freeRunQueueInfo(RunQueueInfo* info) {
  int result = REDISMODULE_OK;
  if (info->run_queue) {
    RedisModule_Free(info->run_queue);
  }
  if (info->threads){
    /* Wait for workers to exit */
    for (int i = 0; i < perqueueThreadPoolSize; i++){
      const int rtn = pthread_join(info->threads[i], NULL);
      if (rtn != 0 ){
          result = REDISMODULE_ERR;
      }
    }
    /* Now free pool structure */
      RedisModule_Free(info->threads);
  }
  RedisModule_Free(info);
  return result;
}

void *RedisAI_Run_ThreadMain(void *arg);

/* Ensure that the the run queue for the device exists.
 * If not, create it. */
int ensureRunQueue(const char* devicestr) {
  int result = REDISMODULE_ERR;

  AI_dictEntry *entry = AI_dictFind(run_queues, devicestr);
  if (entry){
    result = REDISMODULE_OK;
  }
  else{
    RunQueueInfo *run_queue_info = RedisModule_Alloc(sizeof(RunQueueInfo));
    run_queue_info->run_queue = queueCreate();
    pthread_cond_init(&run_queue_info->queue_condition_var, NULL);
    pthread_mutex_init(&run_queue_info->run_queue_mutex, NULL);
    run_queue_info->threads = (pthread_t *)RedisModule_Alloc(sizeof(pthread_t) * perqueueThreadPoolSize);
    /* create threads */
    for (int i = 0; i < perqueueThreadPoolSize; i++){
      if (pthread_create(&(run_queue_info->threads[i]), NULL, RedisAI_Run_ThreadMain, run_queue_info) != 0){
        freeRunQueueInfo(run_queue_info);
        return REDISMODULE_ERR;
      }
    }
    AI_dictAdd(run_queues, (void*)devicestr, (void*)run_queue_info);
    result = REDISMODULE_OK;
  }

  return result;
}

long long ustime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust;
}

mstime_t mstime(void) {
    return ustime()/1000;
}

enum RedisAI_DataFmt {
  REDISAI_DATA_BLOB = 0,
  REDISAI_DATA_VALUES,
  REDISAI_DATA_NONE
};

// ================================

// key type dim1..dimN [BLOB data | VALUES val1..valN]
int RedisAI_TensorSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  const int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_TensorType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  // getting the datatype
  const char* typestr;
  AC_GetString(&ac, &typestr, NULL, 0); 

  size_t datasize = RAI_TensorGetDataSize(typestr);
  if (!datasize){
    return RedisModule_ReplyWithError(ctx, "ERR invalid data type");
  }

  int dims_arg = 3;

  ArgsCursor dac;
  const char* matches[] = {"BLOB", "VALUES"};
  AC_GetSliceUntilMatches(&ac, &dac, 2, matches);

  size_t ndims = dac.argc;
  size_t len = 1;
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (size_t i=0; i<ndims; i++) {
    AC_GetLongLong(&dac, dims+i, 0);
    len *= dims[i];
  }

  if (argc != dims_arg + ndims &&
      argc != dims_arg + ndims + 1 + 1 &&
      argc != dims_arg + ndims + 1 + len) {
    return RedisModule_WrongArity(ctx);
  }

  const int hasdata = !AC_IsAtEnd(&ac);

  const char* fmtstr;
  int datafmt;
  if (hasdata) {
    AC_GetString(&ac, &fmtstr, NULL, 0);
    if (strcasecmp(fmtstr, "BLOB") == 0) {
      datafmt = REDISAI_DATA_BLOB;
    }
    else if (strcasecmp(fmtstr, "VALUES") == 0) {
      datafmt = REDISAI_DATA_VALUES;
    }
    else {
      return RedisModule_ReplyWithError(ctx, "ERR unsupported data format");
    }
  }
  const size_t nbytes = len * datasize;
  size_t datalen;
  const char *data;
  RAI_Tensor *t = RAI_TensorCreate(typestr, dims, ndims, hasdata);
  if (!t){
    return RedisModule_ReplyWithError(ctx, "ERR could not create tensor");
  }
  switch (datafmt){
  case REDISAI_DATA_BLOB:
    AC_GetString(&ac, &data, &datalen, 0);
    if (datalen != nbytes){
      return RedisModule_ReplyWithError(ctx, "ERR data length does not match tensor shape and type");
    }
    RAI_TensorSetData(t, data, datalen);
    break;
  case REDISAI_DATA_VALUES:
    if (argc != len + 4 + ndims){
      return RedisModule_WrongArity(ctx);
    }
    DLDataType datatype = RAI_TensorDataType(t);

    long i;
    if (datatype.code == kDLFloat){
      double val;
      for (i = 0; i < len; i++){
        int ac_ret = AC_GetDouble(&ac, &val, 0);
        if (ac_ret != AC_OK){
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR invalid value");
        }
        int ret = RAI_TensorSetValueFromDouble(t, i, val);
        if (ret == -1){
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
    else{
      long long val;
      for (i = 0; i < len; i++){
        int ac_ret = AC_GetLongLong(&ac, &val, 0);
        if (ac_ret != AC_OK){
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR invalid value");
        }
        int ret = RAI_TensorSetValueFromLongLong(t, i, val);
        if (ret == -1){
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
    break;
  default:
    // default does not require tensor data setting since calloc setted it to 0
    break;
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t);
  RedisModule_CloseKey(key);
  RedisModule_ReplyWithSimpleString(ctx, "OK");
  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

// key [BLOB | VALUES]
int RedisAI_TensorGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 3) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR cannot get tensor from empty key");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_TensorType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  const char* fmtstr;
  int datafmt;
  AC_GetString(&ac, &fmtstr, NULL, 0);
  if (strcasecmp(fmtstr, "BLOB") == 0) {
    datafmt = REDISAI_DATA_BLOB;
  }
  else if (strcasecmp(fmtstr, "VALUES") == 0) {
    datafmt = REDISAI_DATA_VALUES;
  }
  else if (strcasecmp(fmtstr, "META") == 0) {
    datafmt = REDISAI_DATA_NONE;
  }
  else {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR unsupported data format");
  }

  RAI_Tensor *t = RedisModule_ModuleTypeGetValue(key);

  long long ndims = RAI_TensorNumDims(t);
  // (datatype, shape)
  long long resplen = 2;
  if (datafmt == REDISAI_DATA_BLOB || datafmt == REDISAI_DATA_VALUES) {
    resplen += 1;
  }

  RedisModule_ReplyWithArray(ctx, resplen);

  DLDataType dtype = RAI_TensorDataType(t);
  if (dtype.code == kDLFloat) {
    switch (dtype.bits) {
      case 32:
        RedisModule_ReplyWithSimpleString(ctx, "FLOAT");
        break;
      case 64:
        RedisModule_ReplyWithSimpleString(ctx, "DOUBLE");
        break;
      default:
        RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
        break;
    }
  }
  else if (dtype.code == kDLInt) {
    switch (dtype.bits) {
      case 8:
        RedisModule_ReplyWithSimpleString(ctx, "INT8");
        break;
      case 16:
        RedisModule_ReplyWithSimpleString(ctx, "INT16");
        break;
      case 32:
        RedisModule_ReplyWithSimpleString(ctx, "INT32");
        break;
      case 64:
        RedisModule_ReplyWithSimpleString(ctx, "INT64");
        break;
      default:
        RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
        break;
    }
  }
  else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
      case 8:
        RedisModule_ReplyWithSimpleString(ctx, "UINT8");
        break;
      case 16:
        RedisModule_ReplyWithSimpleString(ctx, "UINT16");
        break;
      default:
        RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
        break;
    }
  }
  else {
    assert(0);
  }

  RedisModule_ReplyWithArray(ctx, ndims);
  for (long long i=0; i<ndims; i++) {
    long long dim = RAI_TensorDim(t, i);
    RedisModule_ReplyWithLongLong(ctx, dim);
  }

  if (datafmt == REDISAI_DATA_BLOB) {
    long long size = RAI_TensorByteSize(t);
    char *data = RAI_TensorData(t);

    int ret = RedisModule_ReplyWithStringBuffer(ctx, data, size);

    if (ret != REDISMODULE_OK) {
      RedisModule_CloseKey(key);
      return ret;
    }
  }
  else if (datafmt == REDISAI_DATA_VALUES) {
    long long ndims = RAI_TensorNumDims(t);
    long long len = 1;
    long long i;
    for (i=0; i<ndims; i++) {
      len *= RAI_TensorDim(t, i);
    }

    DLDataType dtype = RAI_TensorDataType(t);

    RedisModule_ReplyWithArray(ctx, len);

    if (dtype.code == kDLFloat) {
      double val;
      for (i=0; i<len; i++) {
        int ret = RAI_TensorGetValueAsDouble(t, i, &val);
        if (!ret) {
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
        }
        RedisModule_ReplyWithDouble(ctx, val);
      }
    }
    else {
      long long val;
      for (i=0; i<len; i++) {
        int ret = RAI_TensorGetValueAsLongLong(t, i, &val);
        if (!ret) {
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
        }
        RedisModule_ReplyWithLongLong(ctx, val);
      }
    }
  }
  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

// ================================

// key backend device [INPUTS name1 name2] [OUTPUTS name1 name2] modelbuf
int RedisAI_ModelSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* bckstr;
  int backend;
  AC_GetString(&ac, &bckstr, NULL, 0); 
  if (strcasecmp(bckstr, "TF") == 0) {
    backend = RAI_BACKEND_TENSORFLOW;
  }
  else if (strcasecmp(bckstr, "TORCH") == 0) {
    backend = RAI_BACKEND_TORCH;
  }
  else if (strcasecmp(bckstr, "ONNX") == 0) {
    backend = RAI_BACKEND_ONNXRUNTIME;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  ArgsCursor optionsac;
  AC_GetSliceToOffset(&ac, &optionsac, argc-2);

  if (optionsac.argc == 0 && backend == RAI_BACKEND_TENSORFLOW) {
    return RedisModule_ReplyWithError(ctx, "Insufficient arguments, INPUTS and OUTPUTS not specified.");
  }

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};
  if (optionsac.argc > 0) {
    if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
      return RedisModule_ReplyWithError(ctx, "INPUTS not specified.");
    }

    const char* matches[] = {"OUTPUTS"};
    AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);

    if (!AC_IsAtEnd(&optionsac)) {
      if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
        return RedisModule_ReplyWithError(ctx, "OUTPUTS not specified.");
      }

      AC_GetSliceToEnd(&optionsac, &outac);
    }
  }

  size_t ninputs = inac.argc;
  const char *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetString(&inac, inputs+i, NULL, 0); 
  }

  size_t noutputs = outac.argc;
  const char *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetString(&outac, outputs+i, NULL, 0); 
  }

  RAI_Model *model = NULL;

  size_t modellen;
  const char *modeldef;
  AC_GetString(&ac, &modeldef, &modellen, 0); 

  RAI_Error err = {0};

  model = RAI_ModelCreate(backend, devicestr, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "Backend %s not loaded, will try loading default backend\n", bckstr);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load %s default backend\n", bckstr);
      int ret = RedisModule_ReplyWithError(ctx, "ERR: could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);
  }

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  // TODO: if backend loaded, make sure there's a queue

  if (ensureRunQueue(devicestr)==REDISMODULE_ERR) {
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR: could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ModelType, model);
  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

// key
int RedisAI_ModelGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR cannot get model from empty key");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Model *mto = RedisModule_ModuleTypeGetValue(key);

  RAI_Error err = {0};

  char *buffer = NULL;
  size_t len = 0;

  RAI_ModelSerialize(mto, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail);
    RAI_ClearError(&err);
    if (*buffer) {
      RedisModule_Free(buffer);
    }
    return ret;
  }

  RedisModule_ReplyWithArray(ctx, 3);

  switch (mto->backend) {
    case REDISAI_BACKEND_TENSORFLOW:
      RedisModule_ReplyWithSimpleString(ctx, "TF");
      break;
    case REDISAI_BACKEND_TORCH:
      RedisModule_ReplyWithSimpleString(ctx, "TORCH");
      break;
    case REDISAI_BACKEND_ONNXRUNTIME:
      RedisModule_ReplyWithSimpleString(ctx, "ONNX");
      break;
  }

  RedisModule_ReplyWithSimpleString(ctx, mto->devicestr);

  RedisModule_ReplyWithStringBuffer(ctx, buffer, len);

  return REDISMODULE_OK;
}

// key
int RedisAI_ModelDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR no model at key");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  RedisModuleString **outkeys;
  RAI_ModelRunCtx *mctx;
  RAI_ScriptRunCtx *sctx;
  int status;
  long long duration_microseconds;
  RAI_Error* err;

};

void RedisAI_FreeRunInfo(RedisModuleCtx *ctx, struct RedisAI_RunInfo *rinfo) {
  if (rinfo->mctx) {
    for(int i = 0 ; i < RAI_ModelRunCtxNumOutputs(rinfo->mctx) ; ++i){
      RedisModule_FreeString(ctx, rinfo->outkeys[i]);
    }
    RedisModule_Free(rinfo->outkeys);
    RAI_ModelRunCtxFree(rinfo->mctx);
  }
  else if (rinfo->sctx) {
    for(int i = 0 ; i < RAI_ScriptRunCtxNumOutputs(rinfo->sctx) ; ++i){
      RedisModule_FreeString(ctx, rinfo->outkeys[i]);
    }
    RedisModule_Free(rinfo->outkeys);
    RAI_ScriptRunCtxFree(rinfo->sctx);
  }

  if (rinfo->err) {
    RAI_ClearError(rinfo->err);
    RedisModule_Free(rinfo->err);
  }

  RedisModule_Free(rinfo);
}

void *RedisAI_RunSession(void *arg) {
  struct RedisAI_RunInfo *rinfo = (struct RedisAI_RunInfo*)arg;
  rinfo->err = RedisModule_Calloc(1, sizeof(RAI_Error));
  const long long start = ustime();
  if (rinfo->mctx) {
    rinfo->status = RAI_ModelRun(rinfo->mctx, rinfo->err);
  }
  else if (rinfo->sctx) {
    rinfo->status = RAI_ScriptRun(rinfo->sctx, rinfo->err);
  }
  rinfo->duration_microseconds = ustime()-start;

  if (rinfo->client != NULL) {
    RedisModule_UnblockClient(rinfo->client, rinfo);
  }
  return NULL;
}

void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo) {
}

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
  RedisModule_Log(ctx, "warning", "Blocked client %p disconnected!", (void*)bc);
}

int RedisAI_Run_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  struct RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);
  
  if (rinfo->status) {
    RedisModule_Log(ctx, "warning", "ERR %s", rinfo->err->detail);
    int ret = RedisModule_ReplyWithError(ctx, rinfo->err->detail_oneline);
    RedisAI_FreeRunInfo(ctx, rinfo);
    return ret;
  }

  size_t num_outputs = 0;
  if (rinfo->mctx) {
    (rinfo->mctx->model->backend_calls)++;
    (rinfo->mctx->model->backend_microseconds)+=rinfo->duration_microseconds;
    num_outputs = RAI_ModelRunCtxNumOutputs(rinfo->mctx);
  }
  else if (rinfo->sctx) {
    (rinfo->sctx->script->backend_calls)++;
    (rinfo->sctx->script->backend_microseconds)+=rinfo->duration_microseconds;
    num_outputs = RAI_ScriptRunCtxNumOutputs(rinfo->sctx);
  }
  for (size_t i=0; i<num_outputs; ++i) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx, rinfo->outkeys[i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(outkey) == RedisAI_TensorType)) {
      RedisModule_CloseKey(outkey);
      RedisAI_FreeRunInfo(ctx, rinfo);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RAI_Tensor *t = NULL;
    if (rinfo->mctx) {
      t = RAI_ModelRunCtxOutputTensor(rinfo->mctx, i);
    }
    else if (rinfo->sctx) {
      t = RAI_ScriptRunCtxOutputTensor(rinfo->sctx, i);
    }
    if (t) {
      RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType, RAI_TensorGetShallowCopy(t));
    }
    RedisModule_CloseKey(outkey);
  }

  // FIXME This crashes Redis, we need to investigate.
  //RedisModule_CloseKey(rinfo->modelkey);

  RedisAI_FreeRunInfo(ctx, rinfo);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

// model key, INPUTS, key1, key2 ... OUTPUTS key1 key2 ...
int RedisAI_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  // 1. clone inputs as needed in the main thread (only the alternative is to lock)
  // 2. spawn the new thread for running the model
  // 3. have reply callback put the data back into the key
  // This way we avoid any race condition. The only gotcha is making sure no one
  // overwrites the model until it's done computing.
  // This means that setModel will decode on a candidate pointer, and will then
  // be picked up on the next round. We also need to signal when it's time to dispose
  // of the old model.
  // The key is having a single thread looping for execution
  if (argc < 3) return RedisModule_WrongArity(ctx);

  if (RedisModule_IsKeysPositionRequest(ctx)) {
    RedisModule_KeyAtPos(ctx, 1);
    for (int i=2; i<argc; i++) {
      const char* arg = RedisModule_StringPtrLen(argv[i], NULL);
      if (strcasecmp(arg, "INPUTS") == 0 || strcasecmp(arg, "OUTPUTS") == 0) {
        continue;
      }
      RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
  }

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR model key is empty");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Model *mto = RedisModule_ModuleTypeGetValue(key);

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};

  if (!AC_AdvanceIfMatch(&ac, "INPUTS")) {
    return RedisModule_ReplyWithError(ctx, "INPUTS not specified.");
  }

  const char* matches[] = {"OUTPUTS"};
  AC_GetSliceUntilMatches(&ac, &inac, 1, matches);

  if (!AC_AdvanceIfMatch(&ac, "OUTPUTS")) {
    return RedisModule_ReplyWithError(ctx, "OUTPUTS not specified.");
  }

  AC_GetSliceToEnd(&ac, &outac);

  size_t ninputs = inac.argc;
  RedisModuleString *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetRString(&inac, inputs+i, 0); 
  }

  size_t noutputs = outac.argc;
  RedisModuleString *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetRString(&outac, outputs+i, 0); 
  }

  if (mto->inputs && array_len(mto->inputs) != ninputs) {
    return RedisModule_ReplyWithError(ctx, "Number of names given as INPUTS during MODELSET and keys given as INPUTS here do not match.");
  }

  if (mto->outputs && array_len(mto->outputs) != noutputs) {
    return RedisModule_ReplyWithError(ctx, "Number of names given as OUTPUTS during MODELSET and keys given as OUTPUTS here do not match.");
  }

  struct RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  rinfo->mctx = RAI_ModelRunCtxCreate(mto);
  rinfo->sctx = NULL;
  rinfo->outkeys = NULL;
  rinfo->err = NULL;

  for (size_t i=0; i<ninputs; i++) {
    RedisModuleKey *argkey = RedisModule_OpenKey(ctx, inputs[i], REDISMODULE_READ);
    int type = RedisModule_KeyType(argkey);
    if (type == REDISMODULE_KEYTYPE_EMPTY) {
      // todo free rinfo, close key
      RedisModule_CloseKey(argkey);
      return RedisModule_ReplyWithError(ctx, "Input key is empty");
    }
    if (!(RedisModule_KeyType(argkey) == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(argkey) == RedisAI_TensorType)) {
      // todo free rinfo, close key
      RedisModule_CloseKey(argkey);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RAI_Tensor *t = RedisModule_ModuleTypeGetValue(argkey);
    RedisModule_CloseKey(argkey);
    // Opname here is passed without copying
    const char* opname = NULL;
    if (mto->inputs) {
      opname = mto->inputs[i];
    }
    if (!RAI_ModelRunCtxAddInput(rinfo->mctx, opname, t)) {
      // todo free rinfo
      return RedisModule_ReplyWithError(ctx, "Input key not found.");
    }
  }

  rinfo->outkeys = RedisModule_Calloc(noutputs, sizeof(RedisModuleString*));
  for (size_t i=0; i<noutputs; i++) {
    // Opname here is passed without copying
    const char* opname = NULL;
    if (mto->outputs) {
      opname = mto->outputs[i];
    }
    if (!RAI_ModelRunCtxAddOutput(rinfo->mctx, opname)) {
      // todo free rinfo
      return RedisModule_ReplyWithError(ctx, "Output key not found.");
    }
    RedisModule_RetainString(ctx, outputs[i]);
    rinfo->outkeys[i] = outputs[i];
  }

  //  RedisModule_AbortBlock(bc);
  //  return RedisModule_ReplyWithError(ctx, "-ERR Can't start thread");

  AI_dictEntry *entry = AI_dictFind(run_queues, mto->devicestr);
  RunQueueInfo *run_queue_info = NULL;
  if (!entry){
    return RedisModule_ReplyWithError(ctx, "Queue not initialized for device.");
  }
  else{
    run_queue_info = AI_dictGetVal(entry);
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_Run_Reply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  // RedisAI_RunSession(rinfo);
  // RedisAI_FreeRunInfo(ctx, rinfo);
  // return RedisModule_ReplyWithSimpleString(ctx, "foo");
  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

void *RedisAI_Run_ThreadMain(void *arg) {

  RunQueueInfo* run_queue_info = (RunQueueInfo*)arg;

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  while (true){
    int rc = pthread_cond_wait(&run_queue_info->queue_condition_var, &run_queue_info->run_queue_mutex);
    queueItem *item = NULL; 
    while ( (item = queuePop(run_queue_info->run_queue)) != NULL){
      pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
      RedisAI_RunSession(item->value);
      RedisModule_Free(item);
      pthread_mutex_lock(&run_queue_info->run_queue_mutex);
    }
    
  }
}

// script key, fnname, INPUTS, key1, key2 ... OUTPUTS, key1, key2 ...
int RedisAI_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  if (RedisModule_IsKeysPositionRequest(ctx)) {
    RedisModule_KeyAtPos(ctx, 1);
    for (int i=3; i<argc; i++) {
      const char* arg = RedisModule_StringPtrLen(argv[i], NULL);
      if (strcasecmp(arg, "INPUTS") == 0 || strcasecmp(arg, "OUTPUTS") == 0) {
        continue;
      }
      RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
  }

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  // TODO we run synchronously for now, but we could have
  // - A: a separate thread and queue for scripts
  // - B: the same thread and queue for models and scripts
  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR script key is empty");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  const char* fnname;
  AC_GetString(&ac, &fnname, NULL, 0); 

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};

  if (!AC_AdvanceIfMatch(&ac, "INPUTS")) {
    return RedisModule_ReplyWithError(ctx, "INPUTS not specified.");
  }

  const char* matches[] = {"OUTPUTS"};
  AC_GetSliceUntilMatches(&ac, &inac, 1, matches);

  if (!AC_AdvanceIfMatch(&ac, "OUTPUTS")) {
    return RedisModule_ReplyWithError(ctx, "OUTPUTS not specified.");
  }

  AC_GetSliceToEnd(&ac, &outac);

  size_t ninputs = inac.argc;
  RedisModuleString *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetRString(&inac, inputs+i, 0); 
  }

  size_t noutputs = outac.argc;
  RedisModuleString *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetRString(&outac, outputs+i, 0); 
  }

  RAI_Script *sto = RedisModule_ModuleTypeGetValue(key);

  RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(sto, fnname);

  RedisModuleString **outkeys;

  for (size_t i=0; i<ninputs; i++) {
    RedisModuleKey *argkey = RedisModule_OpenKey(ctx, inputs[i], REDISMODULE_READ);
    int type = RedisModule_KeyType(argkey);
    if (type == REDISMODULE_KEYTYPE_EMPTY) {
      RedisModule_CloseKey(argkey);
      return RedisModule_ReplyWithError(ctx, "Input key is empty");
    }
    if (!(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(argkey) == RedisAI_TensorType)) {
      RedisModule_CloseKey(argkey);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RAI_Tensor *t = RedisModule_ModuleTypeGetValue(argkey);
    RedisModule_CloseKey(argkey);
    if (!RAI_ScriptRunCtxAddInput(sctx, t)) {
      return RedisModule_ReplyWithError(ctx, "Input key not found.");
    }
  }

  outkeys = RedisModule_Calloc(noutputs, sizeof(RedisModuleString*));
  for (size_t i=0; i<noutputs; i++) {
    if (!RAI_ScriptRunCtxAddOutput(sctx)) {
      return RedisModule_ReplyWithError(ctx, "Output key not found.");
    }
    RedisModule_RetainString(ctx, outputs[i]);
    outkeys[i] = outputs[i];
  }

  struct RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  rinfo->mctx = NULL;
  rinfo->sctx = sctx;
  rinfo->outkeys = outkeys;
  rinfo->err = NULL;
  AI_dictEntry *entry = AI_dictFind(run_queues, sto->devicestr);
  RunQueueInfo *run_queue_info = NULL;
  if (!entry){
    return RedisModule_ReplyWithError(ctx, "Queue not initialized for device.");
  }
  else{
    run_queue_info = AI_dictGetVal(entry);
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_Run_Reply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  RedisModule_ReplicateVerbatim(ctx);
 
  // RAI_Error err = {0};
  // int ret = RAI_ScriptRun(sctx, &err);

  // if (err.code != RAI_OK) {
  //   #ifdef RAI_PRINT_BACKEND_ERRORS
  //   printf("ERR: %s\n", err.detail);
  //   #endif
  //   int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
  //   RAI_ClearError(&err);
  //   return ret;
  // }

  // for (size_t i=0; i<RAI_ScriptRunCtxNumOutputs(sctx); ++i) {
  //   RedisModuleKey *outkey = RedisModule_OpenKey(ctx, outkeys[i],
  //                                                REDISMODULE_READ|REDISMODULE_WRITE);
  //   int type = RedisModule_KeyType(outkey);
  //   if (type != REDISMODULE_KEYTYPE_EMPTY &&
  //       !(type == REDISMODULE_KEYTYPE_MODULE &&
  //         RedisModule_ModuleTypeGetType(outkey) == RedisAI_TensorType)) {
  //     RedisModule_CloseKey(outkey);
  //     return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  //   }
  //   RAI_Tensor *t = RAI_ScriptRunCtxOutputTensor(sctx, i);
  //   if (t) {
  //     RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType, RAI_TensorGetShallowCopy(t));
  //   }
  //   RedisModule_CloseKey(outkey);
  // }

  // RAI_ScriptRunCtxFree(sctx);

  return REDISMODULE_OK;
}

// key
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR cannot get script from empty key");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Script *sto = RedisModule_ModuleTypeGetValue(key);

  RedisModule_ReplyWithArray(ctx, 2);
  RedisModule_ReplyWithSimpleString(ctx, sto->devicestr);
  RedisModule_ReplyWithSimpleString(ctx, sto->scriptdef);

  return REDISMODULE_OK;
}

// key
int RedisAI_ScriptDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr, REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR no script at key");
  }
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

// key device scriptdef
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 4) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  RAI_Script *script = NULL;

  size_t scriptlen;
  const char *scriptdef;
  AC_GetString(&ac, &scriptdef, &scriptlen, 0); 

  RAI_Error err = {0};
  script = RAI_ScriptCreate( devicestr, scriptdef, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "Backend TORCH not loaded, will try loading default backend\n");
    int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load TORCH default backend\n");
      int ret = RedisModule_ReplyWithError(ctx, "ERR: could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    script = RAI_ScriptCreate(devicestr, scriptdef, &err);
  }

  if (err.code != RAI_OK){
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  if (ensureRunQueue(devicestr)==REDISMODULE_ERR) {
    RAI_ScriptFree(script, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR: could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ScriptType, script);
  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

int RedisAI_Config_LoadBackend(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* backend;
  AC_GetString(&ac, &backend, NULL, 0); 

  const char* path;
  AC_GetString(&ac, &path, NULL, 0); 

  int ret;
  if (strcasecmp(backend, "TF") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, path);
  }
  else if (strcasecmp(backend, "TORCH") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, path);
  }
  else if (strcasecmp(backend, "ONNX") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, path);
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  if (ret == REDISMODULE_OK) {
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
  }

  return RedisModule_ReplyWithError(ctx, "ERR error loading backend");
}

int RedisAI_Config_BackendsPath(RedisModuleCtx *ctx, const char *path) {
  RedisModule_AutoMemory(ctx);

  if (RAI_BackendsPath != NULL) {
    RedisModule_Free(RAI_BackendsPath);
  }
  RAI_BackendsPath = RedisModule_Strdup(path);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int RedisAI_Config_QueueThreads(RedisModuleString *queueThreadsString) {
  int result = RedisModule_StringToLongLong(queueThreadsString, &perqueueThreadPoolSize);
  // make sure the number of threads is a positive integer
  // if not set the value to the default 
  if (result == REDISMODULE_OK && perqueueThreadPoolSize < 1 ){
    perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
    result = REDISMODULE_ERR;
  }
  return result;
}

int RedisAI_Config_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* subcommand;
  AC_GetString(&ac, &subcommand, NULL, 0); 

  if (strcasecmp(subcommand, "LOADBACKEND") == 0) {
    return RedisAI_Config_LoadBackend(ctx, argv + 1, argc - 1);
  }

  if (strcasecmp(subcommand, "BACKENDSPATH") == 0) {
    if (argc > 2) {
      return RedisAI_Config_BackendsPath(ctx, RedisModule_StringPtrLen(argv[2], NULL));
    } else {
      return RedisModule_ReplyWithError(ctx, "ERR BACKENDSPATH: missing path argument");
    }
  }

  return RedisModule_ReplyWithError(ctx, "ERR unsupported subcommand");
}

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, ctx) \
  if (RedisModule_ExportSharedAPI) {\
    if (RedisModule_ExportSharedAPI(ctx, "RedisAI_" #name, RAI_ ## name) != REDISMODULE_OK) {\
      RedisModule_Log(ctx, "warning", "Could not register RedisAI_%s", #name);\
      return REDISMODULE_ERR;\
    }\
  }

static int RAI_GetLLAPIVersion(){
  return REDISAI_LLAPI_VERSION;
}

static int RedisAI_RegisterApi(RedisModuleCtx* ctx) {

  if (!RedisModule_ExportSharedAPI) {
    RedisModule_Log(ctx, "warning", "Redis version does not support SharedAPI; running without exposing C API to other modules.");
  }

  REGISTER_API(GetLLAPIVersion, ctx);

  REGISTER_API(TensorCreate, ctx);
  REGISTER_API(TensorGetDataSize, ctx);
  REGISTER_API(TensorFree, ctx);
  REGISTER_API(TensorSetData, ctx);
  REGISTER_API(TensorSetValueFromLongLong, ctx);
  REGISTER_API(TensorSetValueFromDouble, ctx);
  REGISTER_API(TensorGetValueAsDouble, ctx);
  REGISTER_API(TensorGetValueAsLongLong, ctx);
  REGISTER_API(TensorGetShallowCopy, ctx);
  REGISTER_API(TensorNumDims, ctx);
  REGISTER_API(TensorDim, ctx);
  REGISTER_API(TensorByteSize, ctx);
  REGISTER_API(TensorData, ctx);

  REGISTER_API(ModelCreate, ctx);
  REGISTER_API(ModelFree, ctx);
  REGISTER_API(ModelRunCtxCreate, ctx);
  REGISTER_API(ModelRunCtxAddInput, ctx);
  REGISTER_API(ModelRunCtxAddOutput, ctx);
  REGISTER_API(ModelRunCtxNumOutputs, ctx);
  REGISTER_API(ModelRunCtxOutputTensor, ctx);
  REGISTER_API(ModelRunCtxFree, ctx);
  REGISTER_API(ModelRun, ctx);
  REGISTER_API(ModelSerialize, ctx);
  REGISTER_API(ModelGetShallowCopy, ctx);

  REGISTER_API(ScriptCreate, ctx);
  REGISTER_API(ScriptFree, ctx);
  REGISTER_API(ScriptRunCtxCreate, ctx);
  REGISTER_API(ScriptRunCtxAddInput, ctx);
  REGISTER_API(ScriptRunCtxAddOutput, ctx);
  REGISTER_API(ScriptRunCtxNumOutputs, ctx);
  REGISTER_API(ScriptRunCtxOutputTensor, ctx);
  REGISTER_API(ScriptRunCtxFree, ctx);
  REGISTER_API(ScriptRun, ctx);
  REGISTER_API(ScriptGetShallowCopy, ctx);

  return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "ai", RAI_ENC_VER, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  int flags = RedisModule_GetContextFlags(ctx);
  if (flags & REDISMODULE_CTX_FLAGS_AOF) {
    RedisModule_Log(ctx, "warning", "ERR: AOF currently unsupported\r\n");
#ifndef RAI_OVERRIDE_AOF_CHECK
    return REDISMODULE_ERR;
#endif
  }

  RAI_BackendsPath = NULL;

  if(RedisAI_RegisterApi(ctx) != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_TensorInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ModelInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize model dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ScriptInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize script dt\r\n");
    return REDISMODULE_ERR;
  }

  if (RedisModule_CreateCommand(ctx, "ai.tensorset", RedisAI_TensorSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.tensorget", RedisAI_TensorGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelset", RedisAI_ModelSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelget", RedisAI_ModelGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modeldel", RedisAI_ModelDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelrun", RedisAI_ModelRun_RedisCommand, "write deny-oom getkeys-api", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptset", RedisAI_ScriptSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptget", RedisAI_ScriptGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptdel", RedisAI_ScriptDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptrun", RedisAI_ScriptRun_RedisCommand, "write deny-oom getkeys-api", 4, 4, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.config", RedisAI_Config_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (argc > 0 && argc % 2 != 0) {
    RedisModule_Log(ctx, "warning", "Even number of arguments provided to module. Please provide arguments as KEY VAL pairs.");
  }

  // need BACKENDSPATH set up before loading specific backends
  for (int i=0; i<argc/2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2*i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2*i + 1], NULL);

    int ret = REDISMODULE_OK;
    if (strcasecmp(key, "BACKENDSPATH") == 0) {
      ret = RedisAI_Config_BackendsPath(ctx, val);
    }
  }

  for (int i=0; i<argc/2; i++) {
    const char *key = RedisModule_StringPtrLen(argv[2*i], NULL);
    const char *val = RedisModule_StringPtrLen(argv[2*i + 1], NULL);

    int ret = REDISMODULE_OK;
    if (strcasecmp(key, "TF") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TENSORFLOW, val);
    }
    else if (strcasecmp(key, "TORCH") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TORCH, val);
    }
    else if (strcasecmp(key, "ONNX") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_ONNXRUNTIME, val);
    }
    // enable configuring the main thread to create a fixed number of worker threads up front per device.
    // by default we'll use 1
    else if (strcasecmp(key, "THREADS_PER_QUEUE") == 0) {
      ret = RedisAI_Config_QueueThreads(argv[2*i + 1]);
      if (ret == REDISMODULE_OK){
        char *buffer = RedisModule_Alloc((3 + strlen(REDISAI_INFOMSG_THREADS_PER_QUEUE) + strlen(val)) * sizeof(*buffer));
        sprintf(buffer, "%s: %s", REDISAI_INFOMSG_THREADS_PER_QUEUE, val);
        RedisModule_Log(ctx, "verbose", buffer);
        RedisModule_Free(buffer);
      }
    }
    else if (strcasecmp(key, "BACKENDSPATH") == 0) {
      // aleady taken care of
    } else {
      ret = REDISMODULE_ERR;
    }

    if (ret == REDISMODULE_ERR) {
      char* buffer = RedisModule_Alloc((4 + strlen(REDISAI_ERRORMSG_PROCESSING_ARG) + strlen(key) + strlen(val)) * sizeof(*buffer));
      sprintf(buffer, "%s: %s %s", REDISAI_ERRORMSG_PROCESSING_ARG, key, val);
      RedisModule_Log(ctx, "warning", buffer);
      RedisModule_Free(buffer);
    }
  }

  run_queues = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);

  if (ensureRunQueue("CPU") != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "Queue not initialized for device CPU" );
    return REDISMODULE_ERR;
  }
  
  return REDISMODULE_OK;
}

extern AI_dictType AI_dictTypeHeapStrings;
