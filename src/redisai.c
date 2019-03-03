#include "redismodule.h"
#include "tensor.h"
#include "model.h"
#include "script.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

#include "utils/alloc.h"
#include "utils/arr_rm_alloc.h"

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

static pthread_mutex_t runQueueMutex = PTHREAD_MUTEX_INITIALIZER;
static queue *runQueue;

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

  // getting the datatype
  const char* typestr = RedisModule_StringPtrLen(argv[2], NULL);
  size_t datasize = RAI_TensorGetDataSize(typestr);
  if (!datasize){
    return RedisModule_ReplyWithError(ctx, "ERR invalid data type");
  }

  int dims_arg = 3;

  long long ndims = 0;
  long long len = 1;
  long long *dims_ = RedisModule_PoolAlloc(ctx, (argc - dims_arg) * sizeof(long long));
  for (long long i=0; i<argc-dims_arg; i++) {
    if (RedisModule_StringToLongLong(argv[3+i], dims_+i) != REDISMODULE_OK) {
      break;
    }
    ndims++;
    len *= dims_[i];
  }

  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  memcpy(dims, dims_, ndims * sizeof(long long));

  if (argc != dims_arg + ndims &&
      argc != dims_arg + ndims + 1 + 1 &&
      argc != dims_arg + ndims + 1 + len) {
    return RedisModule_WrongArity(ctx);
  }

  int datafmt_arg = dims_arg + ndims;
  int hasdata = argc > datafmt_arg;

  const char* fmtstr;
  int datafmt;
  if (hasdata) {
    fmtstr = RedisModule_StringPtrLen(argv[datafmt_arg], NULL);
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

  size_t nbytes = len * datasize;

  size_t datalen;
  const char* data;
  if (hasdata) {
    data = RedisModule_StringPtrLen(argv[datafmt_arg + 1], &datalen);
    if (datafmt == REDISAI_DATA_BLOB && datalen != nbytes) {
      return RedisModule_ReplyWithError(ctx, "ERR data length does not match tensor shape and type");
    }
  }

  if (hasdata && datafmt == REDISAI_DATA_VALUES && argc != len + 4 + ndims) {
    return RedisModule_WrongArity(ctx);
  }

  RAI_Tensor* t = RAI_TensorCreate(typestr, dims, ndims);
  if (!t) {
    return RedisModule_ReplyWithError(ctx, "ERR could not create tensor");
  }

  DLDataType datatype = RAI_TensorDataType(t);

  if (hasdata && datafmt == REDISAI_DATA_BLOB) {
    RAI_TensorSetData(t, data, datalen);
  }
  else if (hasdata && datafmt == REDISAI_DATA_VALUES) {
    long i;
    if (datatype.code == kDLFloat) {
      double val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToDouble(argv[datafmt_arg + 1 + i], &val) != REDISMODULE_OK)) {
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR invalid value");
        }
        int ret = RAI_TensorSetValueFromDouble(t, i, val);
        if (ret == -1) {
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
    else {
      long long val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToLongLong(argv[datafmt_arg + 1 + i], &val) != REDISMODULE_OK)) {
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR invalid value");
        }
        int ret = RAI_TensorSetValueFromLongLong(t, i, val);
        if (ret == -1) {
          RAI_TensorFree(t);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_TensorType)) {
    RAI_TensorFree(t);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
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

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (!(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_TensorType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  const char* fmtstr;
  int datafmt;
  fmtstr = RedisModule_StringPtrLen(argv[2], NULL);
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
        RedisModule_ReplyWithSimpleString(ctx, "INT8");
        break;
      case 16:
        RedisModule_ReplyWithSimpleString(ctx, "INT16");
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

  const char* bckstr;
  int backend;
  bckstr = RedisModule_StringPtrLen(argv[2], NULL);
  if (strcasecmp(bckstr, "TF") == 0) {
    backend = RAI_BACKEND_TENSORFLOW;
  }
  else if (strcasecmp(bckstr, "TORCH") == 0) {
    backend = RAI_BACKEND_TORCH;
  }
  else if (strcasecmp(bckstr, "ORT") == 0) {
    backend = RAI_BACKEND_ONNXRUNTIME;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  const char* devicestr;
  int device;
  devicestr = RedisModule_StringPtrLen(argv[3], NULL);
  if (strcasecmp(devicestr, "CPU") == 0) {
    device = RAI_DEVICE_CPU;
  }
  else if (strcasecmp(devicestr, "GPU") == 0) {
    device = RAI_DEVICE_GPU;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported device");
  }

  size_t argidx = 4;

  size_t remaining_args = argc - argidx;

  if (remaining_args > 1000) {
    return RedisModule_ReplyWithError(ctx, "ERR too many input or output arguments");
  }

  const char *inputs[remaining_args];
  const char *outputs[remaining_args];
  size_t ninputs = 0;
  size_t noutputs = 0;

  const char *name, *option;

  option = RedisModule_StringPtrLen(argv[argidx], NULL);

  if (strcasecmp(option, "INPUTS") == 0) {
    argidx++;
    for (size_t i=0; i<argc-argidx-1; i++) {
      name = RedisModule_StringPtrLen(argv[argidx + i], NULL);
      if (strcasecmp(name, "OUTPUTS") == 0) {
        break;
      }
      inputs[i] = name;
      ninputs++;
    }
    argidx += ninputs;
  }

  option = RedisModule_StringPtrLen(argv[argidx], NULL);

  if (strcasecmp(option, "OUTPUTS") == 0) {
    argidx++;
    for (size_t i=0; i<argc-argidx-1; i++) {
      outputs[i] = RedisModule_StringPtrLen(argv[argidx + i], NULL);
      noutputs++;
    }
    argidx += noutputs;
  }

  RAI_Model *model = NULL;

  size_t modellen;
  const char *modeldef = RedisModule_StringPtrLen(argv[argidx], &modellen);

  RAI_Error err = RAI_InitError();

  model = RAI_ModelCreate(backend, device, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
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

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
  if (!(RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Model *mto = RedisModule_ModuleTypeGetValue(key);

  RAI_Error err = RAI_InitError();

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

  RedisModule_ReplyWithArray(ctx, 2);
  //RedisModule_ReplyWithSimpleString(ctx, mto->backend);
  //RedisModule_ReplyWithSimpleString(ctx, mto->device);
  RedisModule_ReplyWithLongLong(ctx, mto->backend);
  RedisModule_ReplyWithLongLong(ctx, mto->device);
  RedisModule_ReplyWithStringBuffer(ctx, buffer, len);

  return REDISMODULE_OK;
}

struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  RedisModuleString **outkeys;
  RAI_ModelRunCtx* mctx;
  int status;
};

void RedisAI_FreeRunInfo(RedisModuleCtx *ctx, struct RedisAI_RunInfo *rinfo) {
  for(int i = 0 ; i < RAI_ModelRunCtxNumOutputs(rinfo->mctx) ; ++i){
    RedisModule_FreeString(ctx, rinfo->outkeys[i]);
  }
  RedisModule_Free(rinfo->outkeys);

  RAI_ModelRunCtxFree(rinfo->mctx);

  RedisModule_Free(rinfo);
}

void *RedisAI_RunSession(void *arg) {
  struct RedisAI_RunInfo *rinfo = (struct RedisAI_RunInfo*)arg;

  RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(rinfo->client);

  RAI_Error err = RAI_InitError();

  mstime_t start = mstime();
  rinfo->status = RAI_ModelRun(rinfo->mctx, &err);
  mstime_t end = mstime();

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return NULL;
  }

  RedisModule_ThreadSafeContextLock(ctx);
  RedisModule_Log(ctx, "notice", "RAI_ModelRun took %fs", (end - start) / 1000.0);
  RedisModule_ThreadSafeContextUnlock(ctx);

  RedisModule_UnblockClient(rinfo->client, rinfo);

  RedisModule_FreeThreadSafeContext(ctx);

  return NULL;
}

//void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
//  RedisModule_Log(ctx,"warning","Blocked client %p disconnected!", (void*)bc);
//
//  // TODO: clean up
//}

int RedisAI_Run_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  struct RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

  if (rinfo->status) {
    int ret = RedisModule_ReplyWithError(ctx, "model run failed");
    RedisAI_FreeRunInfo(ctx, rinfo);
    return ret;
  }

  for (size_t i=0; i<RAI_ModelRunCtxNumOutputs(rinfo->mctx); ++i) {
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
    RAI_Tensor *t = RAI_ModelRunCtxOutputTensor(rinfo->mctx, i);
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

  RedisModule_AutoMemory(ctx);

  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
  if (!(RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Model *mto = RedisModule_ModuleTypeGetValue(key);

  struct RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  rinfo->mctx = RAI_ModelRunCtxCreate(mto);
  rinfo->outkeys = NULL;

  long long argidx = 2;

  while (argidx < argc-1) {
    bool in_inputs;
    const char* section;
    section = RedisModule_StringPtrLen(argv[argidx], NULL);
    if (strcasecmp(section, "INPUTS") == 0) {
      in_inputs = true;
    }
    else if (strcasecmp(section, "OUTPUTS") == 0) {
      in_inputs = false;
    }

    RedisModuleString** keys = array_new(RedisModuleString*, argc - argidx);

    // if (argidx + 1 + nitems > argc) {
    //   return RedisModule_ReplyWithError(ctx, "Insufficient inputs in command.");
    // }
 
    for (int idx=1; idx<argc - argidx; idx++) {
      const char *keyptr = RedisModule_StringPtrLen(argv[argidx + idx], NULL);
      if (strcasecmp(keyptr, "OUTPUTS") == 0) {
        break;
      }
      array_append(keys, argv[argidx + idx]);
    }

    long long nitems = array_len(keys);

    int argoffset = nitems + 1;

    if (argidx + argoffset > argc) {
      return RedisModule_ReplyWithError(ctx, "Insufficient inputs in command.");
    }

    if (in_inputs) {
      for (int i = 0; i < nitems; i++) {
        RedisModuleKey *argkey = RedisModule_OpenKey(ctx, keys[i], REDISMODULE_READ);
        if (!(RedisModule_KeyType(argkey) == REDISMODULE_KEYTYPE_MODULE &&
              RedisModule_ModuleTypeGetType(argkey) == RedisAI_TensorType)) {
          // todo free rinfo
          RedisModule_CloseKey(argkey);
          return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
        }
        RAI_Tensor *t = RedisModule_ModuleTypeGetValue(argkey);
        RedisModule_CloseKey(argkey);
        // Opname here is passed without copying
        const char* opname = NULL;
        if (mto->inputs) {
          size_t ninputs = array_len(mto->inputs);
          if (i >= ninputs) {
            return RedisModule_ReplyWithError(ctx, "Names and inputs do not match.");
          }
          opname = mto->inputs[i];
        }
        if (!RAI_ModelRunCtxAddInput(rinfo->mctx, opname, t)) {
          // todo free rinfo
          return RedisModule_ReplyWithError(ctx, "Input key not found.");
        }
      }
    }
    else {
      rinfo->outkeys = RedisModule_Calloc(nitems, sizeof(RedisModuleString*));
      for (int i = 0; i < nitems; i++) {
        // Opname here is passed without copying
        const char* opname = NULL;
        if (mto->outputs) {
          size_t noutputs = array_len(mto->outputs);
          if (i >= noutputs) {
            return RedisModule_ReplyWithError(ctx, "Names and outputs do not match.");
          }
          opname = mto->outputs[i];
        }
        if (!RAI_ModelRunCtxAddOutput(rinfo->mctx, opname)) {
          // todo free rinfo
          return RedisModule_ReplyWithError(ctx, "Output key not found.");
        }
        RedisModule_RetainString(ctx, keys[i]);
        rinfo->outkeys[i] = keys[i];
      }
    }

    array_free(keys);

    argidx += argoffset;
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_Run_Reply, NULL, NULL, 0);

  //  RedisModule_AbortBlock(bc);
  //  return RedisModule_ReplyWithError(ctx, "-ERR Can't start thread");

  pthread_mutex_lock(&runQueueMutex);
  queuePush(runQueue, rinfo);
  pthread_mutex_unlock(&runQueueMutex);

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

void *RedisAI_Run_ThreadMain(void *arg) {

  while(1) {
    pthread_mutex_lock(&runQueueMutex);
    queueItem *item = queuePop(runQueue);
    pthread_mutex_unlock(&runQueueMutex);

    if (item) {
      RedisAI_RunSession(item->value);
    }

    // release item (note: the callback does it; verify)
    // unblock (the callback does it)

    usleep(1000);
  }
}

int RedisAI_StartRunThread() {
  pthread_t tid;

  runQueue = queueCreate();

  if (pthread_create(&tid, NULL, RedisAI_Run_ThreadMain, NULL) != 0) {
    return REDISMODULE_ERR;
  }

  return REDISMODULE_OK;
}

// script key, INPUTS, key1, key2 ... OUTPUTS, key1, key2 ...
int RedisAI_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  // TODO we run synchronously for now, but we could have
  // - A: a separate thread and queue for scripts
  // - B: the same thread and queue for models and scripts
  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
  if (!(RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  const char* fnname;
  fnname = RedisModule_StringPtrLen(argv[2], NULL);
 
  RAI_Script *sto = RedisModule_ModuleTypeGetValue(key);

  RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(sto);

  size_t fnname_len = strlen(fnname);
  sctx->fnname = RedisModule_Calloc(fnname_len, sizeof(char));
  memcpy(sctx->fnname, fnname, fnname_len);

  RedisModuleString **outkeys;

  long long argidx = 3;

  while (argidx < argc-1) {
    bool in_inputs;
    const char* section;
    section = RedisModule_StringPtrLen(argv[argidx], NULL);
    if (strcasecmp(section, "INPUTS") == 0) {
      in_inputs = true;
    }
    else if (strcasecmp(section, "OUTPUTS") == 0) {
      in_inputs = false;
    }

    RedisModuleString** keys = array_new(RedisModuleString*, argc - argidx);

    for (int idx=1; idx<argc - argidx; idx++) {
      const char *keyptr = RedisModule_StringPtrLen(argv[argidx + idx], NULL);
      if (strcasecmp(keyptr, "OUTPUTS") == 0) {
        break;
      }
      array_append(keys, argv[argidx + idx]);
    }

    long long nitems = array_len(keys);

    int argoffset = nitems + 1;

    if (in_inputs) {
      for (int i = 0; i < nitems; i++) {
        RedisModuleKey *argkey = RedisModule_OpenKey(ctx, keys[i], REDISMODULE_READ);
        if (!(RedisModule_KeyType(argkey) == REDISMODULE_KEYTYPE_MODULE &&
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
    }
    else {
      outkeys = RedisModule_Calloc(nitems, sizeof(RedisModuleString*));
      for (int i = 0; i < nitems; i++) {
        if (!RAI_ScriptRunCtxAddOutput(sctx)) {
          return RedisModule_ReplyWithError(ctx, "Output key not found.");
        }
        RedisModule_RetainString(ctx, keys[i]);
        outkeys[i] = keys[i];
      }
    }

    array_free(keys);

    argidx += argoffset;
  }

  RAI_Error err = RAI_InitError();
  int ret = RAI_ScriptRun(sctx, &err);

  if (err.code != RAI_OK) {
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  for (size_t i=0; i<RAI_ScriptRunCtxNumOutputs(sctx); ++i) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx, outkeys[i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        !(type == REDISMODULE_KEYTYPE_MODULE &&
          RedisModule_ModuleTypeGetType(outkey) == RedisAI_TensorType)) {
      RedisModule_CloseKey(outkey);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RAI_Tensor *t = RAI_ScriptRunCtxOutputTensor(sctx, i);
    if (t) {
      RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType, RAI_TensorGetShallowCopy(t));
    }
    RedisModule_CloseKey(outkey);
  }

  RAI_ScriptRunCtxFree(sctx);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

// key
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
  if (!(RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Script *sto = RedisModule_ModuleTypeGetValue(key);

  RedisModule_ReplyWithArray(ctx, 2);
  // RedisModule_ReplyWithSimpleString(ctx, sto->device);
  RedisModule_ReplyWithLongLong(ctx, sto->device);
  RedisModule_ReplyWithSimpleString(ctx, sto->scriptdef);

  return REDISMODULE_OK;
}

// key device scriptdef
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 4) return RedisModule_WrongArity(ctx);

  const char* devicestr;
  int device;
  devicestr = RedisModule_StringPtrLen(argv[2], NULL);
  if (strcasecmp(devicestr, "CPU") == 0) {
    device = RAI_DEVICE_CPU;
  }
  else if (strcasecmp(devicestr, "GPU") == 0) {
    device = RAI_DEVICE_GPU;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported device");
  }

  RAI_Script *script = NULL;

  size_t scriptlen;
  const char *scriptdef = RedisModule_StringPtrLen(argv[3], &scriptlen);

  RAI_Error err = RAI_InitError();
  script = RAI_ScriptCreate(device, scriptdef, &err);

  if (err.code != RAI_OK){
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
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

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, registerApiCallback) \
  if(registerApiCallback("RedisAI_" #name, RAI_ ## name)){\
      printf("could not register RedisAI_" #name "\r\n");\
      return false;\
  }

static bool RediAI_RegisterApi(int (*registerApiCallback)(const char *funcname, void *funcptr)){
  REGISTER_API(TensorCreate, registerApiCallback);
  REGISTER_API(TensorGetDataSize, registerApiCallback);
  REGISTER_API(TensorFree, registerApiCallback);
  REGISTER_API(TensorSetData, registerApiCallback);
  REGISTER_API(TensorSetValueFromLongLong, registerApiCallback);
  REGISTER_API(TensorSetValueFromDouble, registerApiCallback);
  REGISTER_API(TensorGetValueAsDouble, registerApiCallback);
  REGISTER_API(TensorGetValueAsLongLong, registerApiCallback);
  REGISTER_API(TensorGetShallowCopy, registerApiCallback);
  REGISTER_API(TensorNumDims, registerApiCallback);
  REGISTER_API(TensorDim, registerApiCallback);
  REGISTER_API(TensorByteSize, registerApiCallback);
  REGISTER_API(TensorData, registerApiCallback);

  REGISTER_API(ModelCreate, registerApiCallback);
  REGISTER_API(ModelFree, registerApiCallback);
  REGISTER_API(ModelRunCtxCreate, registerApiCallback);
  REGISTER_API(ModelRunCtxAddInput, registerApiCallback);
  REGISTER_API(ModelRunCtxAddOutput, registerApiCallback);
  REGISTER_API(ModelRunCtxNumOutputs, registerApiCallback);
  REGISTER_API(ModelRunCtxOutputTensor, registerApiCallback);
  REGISTER_API(ModelRunCtxFree, registerApiCallback);
  REGISTER_API(ModelRun, registerApiCallback);
  REGISTER_API(ModelSerialize, registerApiCallback);
  REGISTER_API(ModelGetShallowCopy, registerApiCallback);

  REGISTER_API(ScriptCreate, registerApiCallback);
  REGISTER_API(ScriptFree, registerApiCallback);
  REGISTER_API(ScriptRunCtxCreate, registerApiCallback);
  REGISTER_API(ScriptRunCtxAddInput, registerApiCallback);
  REGISTER_API(ScriptRunCtxAddOutput, registerApiCallback);
  REGISTER_API(ScriptRunCtxNumOutputs, registerApiCallback);
  REGISTER_API(ScriptRunCtxOutputTensor, registerApiCallback);
  REGISTER_API(ScriptRunCtxFree, registerApiCallback);
  REGISTER_API(ScriptRun, registerApiCallback);
  REGISTER_API(ScriptGetShallowCopy, registerApiCallback);

  return true;
}

int moduleRegisterApi(const char *funcname, void *funcptr);

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "ai", 100, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  if(!RediAI_RegisterApi(moduleRegisterApi)){
    RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_TensorInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt");
    return REDISMODULE_ERR;
  }

  if(!RAI_ModelInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize model dt");
    return REDISMODULE_ERR;
  }

  if(!RAI_ScriptInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize script dt");
    return REDISMODULE_ERR;
  }

  if (RedisModule_CreateCommand(ctx, "ai.tensorset", RedisAI_TensorSet_RedisCommand, "write", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.tensorget", RedisAI_TensorGet_RedisCommand, "readonly", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;


  if (RedisModule_CreateCommand(ctx, "ai.modelset", RedisAI_ModelSet_RedisCommand, "write", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelget", RedisAI_ModelGet_RedisCommand, "readonly", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelrun", RedisAI_ModelRun_RedisCommand, "write", 2, -1, 2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptset", RedisAI_ScriptSet_RedisCommand, "write", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptget", RedisAI_ScriptGet_RedisCommand, "readonly", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptrun", RedisAI_ScriptRun_RedisCommand, "write", 2, -1, 2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisAI_StartRunThread() == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}

