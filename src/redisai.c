#include "redismodule.h"
#include "tensor.h"
#include "graph.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

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

  if ((queue = RedisModule_Alloc(sizeof(*queue))) == NULL)
    return NULL;

  queue->front = queue->back = NULL;
  queue->len = 0;
  queue->free = NULL;
  return queue;
}

void queuePush(queue *queue, void *value) {
  queueItem *item;

  if ((item = RedisModule_Alloc(sizeof(*item))) == NULL)
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

// key type ndims dim1..dimN [BLOB data | VALUES val1..valN]
int RedisAI_Set_Tensor_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 5) return RedisModule_WrongArity(ctx);

  // getting the datatype
  const char* typestr = RedisModule_StringPtrLen(argv[2], NULL);
  size_t datasize = RAI_TensorGetDataSize(typestr);
  if (!datasize){
    return RedisModule_ReplyWithError(ctx, "ERR invalid data type");
  }

  long long ndims = 0;
  if ((RedisModule_StringToLongLong(argv[3], &ndims) != REDISMODULE_OK)
      || ndims < 0) {
    return RedisModule_ReplyWithError(ctx, "ERR invalid ndims");
  }

  if (argc < 4 + ndims) {
    return RedisModule_WrongArity(ctx);
  }

  long long len = 1;
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (long long i=0; i<ndims; i++) {
    if ((RedisModule_StringToLongLong(argv[4+i], dims+i) != REDISMODULE_OK)
        || dims[i] < 0) {
      return RedisModule_ReplyWithError(ctx, "ERR invalid dims");
    }
    len *= dims[i];
  }

  if (argc != 4 + ndims &&
      argc != 4 + ndims + 1 + 1 &&
      argc != 4 + ndims + 1 + len) {
    return RedisModule_WrongArity(ctx);
  }

  int datafmt_arg = 4 + ndims;
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

  if (hasdata && datafmt == REDISAI_DATA_VALUES && argc != len + 5 + ndims) {
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
          return RedisModule_ReplyWithError(ctx, "ERR invalid val");
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
          return RedisModule_ReplyWithError(ctx, "ERR invalid val");
        }
        int ret = RAI_TensorSetValueFromDouble(t, i, val);
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
      RedisModule_ModuleTypeGetType(key) != RedisAI_TensorType) {
    RAI_TensorFree(t);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

// key [BLOB | VALUES]
int RedisAI_Get_Tensor_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisAI_TensorType) {
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
  // (datatype, ndim, shape, bytesize)
  long long resplen = 4;
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

  RedisModule_ReplyWithLongLong(ctx, ndims);

  RedisModule_ReplyWithArray(ctx, ndims);
  for (long long i=0; i<ndims; i++) {
    long long dim = RAI_TensorDim(t, i);
    RedisModule_ReplyWithLongLong(ctx, dim);
  }

  long long bytesize = RAI_TensorByteSize(t);
  RedisModule_ReplyWithLongLong(ctx, bytesize);

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

// key graphbuf [prefix]
int RedisAI_Set_Graph_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if ((argc != 4) && (argc != 5)) return RedisModule_WrongArity(ctx);

  const char* bckstr;
  int backend;
  bckstr = RedisModule_StringPtrLen(argv[2], NULL);
  if (strcasecmp(bckstr, "TF") == 0) {
    backend = RAI_BACKEND_TENSORFLOW;
  }
  else if (strcasecmp(bckstr, "PT") == 0) {
    backend = RAI_BACKEND_PYTORCH;
  }
  else if (strcasecmp(bckstr, "OR") == 0) {
    backend = RAI_BACKEND_ONNXRUNTIME;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  RAI_Graph *graph = NULL;

  size_t graphlen;
  const char *graphdef = RedisModule_StringPtrLen(argv[3], &graphlen);
  const char *prefix = "";
  if (argc == 5) {
    const char *prefix = RedisModule_StringPtrLen(argv[4], NULL);
  }

  graph = RAI_GraphCreate(prefix, backend, graphdef, graphlen);

  if(graph == NULL){
    return RedisModule_ReplyWithError(ctx, "ERR failed creating the graph");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != RedisAI_GraphType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_GraphType, graph);
  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  RedisModuleString **outkeys;
  RAI_GraphRunCtx* gctx;
  int status;
};

void RedisAI_FreeRunInfo(RedisModuleCtx *ctx, struct RedisAI_RunInfo *rinfo) {
  for(int i = 0 ; i < RAI_RunCtxNumOutputs(rinfo->gctx) ; ++i){
    RedisModule_FreeString(ctx, rinfo->outkeys[i]);
  }
  RedisModule_Free(rinfo->outkeys);

  RAI_RunCtxFree(rinfo->gctx);

  RedisModule_Free(rinfo);
}

void *RedisAI_RunSession(void *arg) {
  struct RedisAI_RunInfo *rinfo = (struct RedisAI_RunInfo*)arg;

  RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(rinfo->client);

  mstime_t start = mstime();
  rinfo->status = RAI_GraphRun(rinfo->gctx);
  mstime_t end = mstime();

  RedisModule_ThreadSafeContextLock(ctx);
  RedisModule_Log(ctx, "notice", "RAI_GraphRun took %fs", (end - start) / 1000.0);
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

  if (!rinfo->status) {
    int ret = RedisModule_ReplyWithError(ctx, "graph run failed");
    RedisAI_FreeRunInfo(ctx, rinfo);
    return ret;
  }

  for (size_t i=0; i<RAI_RunCtxNumOutputs(rinfo->gctx); ++i) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx, rinfo->outkeys[i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        RedisModule_ModuleTypeGetType(outkey) != RedisAI_TensorType) {
      RedisModule_CloseKey(outkey);
      RedisAI_FreeRunInfo(ctx, rinfo);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    RAI_Tensor *t = RAI_RunCtxOutputTensor(rinfo->gctx, i);
    if (t) {
      RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType, RAI_TensorGetShallowCopy(t));
    }
    RedisModule_CloseKey(outkey);
  }

  // FIXME This crashes Redis, we need to investigate.
  //RedisModule_CloseKey(rinfo->graphkey);

  RedisAI_FreeRunInfo(ctx, rinfo);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

// graph key, ninputs, (input key, input name)..., (output key, output name)...
int RedisAI_Run_Graph_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  // 1. clone inputs as needed in the main thread (only the alternative is to lock)
  // 2. spawn the new thread for running the graph
  // 3. have reply callback put the data back into the key
  // This way we avoid any race condition. The only gotcha is making sure no one
  // overwrites the graph until it's done computing.
  // This means that setGraph will decode on a candidate pointer, and will then
  // be picked up on the next round. We also need to signal when it's time to dispose
  // of the old graph.
  // The key is having a single thread looping for execution

  RedisModule_AutoMemory(ctx);

  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
  if (RedisModule_ModuleTypeGetType(key) != RedisAI_GraphType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RAI_Graph *gto = RedisModule_ModuleTypeGetValue(key);

  long long ninputs;
  if ((RedisModule_StringToLongLong(argv[2], &ninputs) != REDISMODULE_OK)
      || ninputs < 0) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR invalid ninputs");
  }

  int pairoffset = 3;
  int npairs = (argc - pairoffset) / 2;
  int noutputs = npairs - ninputs;

  if (npairs < ninputs) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR key/name pairs less than ninputs");
  }

  if ((argc - pairoffset) % 2 != 0) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR odd key/name pairs");
  }

  struct RedisAI_RunInfo *rinfo = RedisModule_Alloc(sizeof(struct RedisAI_RunInfo));
  rinfo->gctx = RAI_RunCtxCreate(gto);

  rinfo->outkeys = RedisModule_Alloc(noutputs*sizeof(RedisModuleString*));

  for (int i=pairoffset; i<argc; i+=2) {
    int isinput = i < pairoffset + 2 * ninputs;

    RedisModuleString* argname = argv[i+1];

    if (isinput) {
      RedisModuleKey *argkey = RedisModule_OpenKey(ctx, argv[i], REDISMODULE_READ);
      if (RedisModule_ModuleTypeGetType(argkey) != RedisAI_TensorType) {
        // todo free rinfo
        RedisModule_CloseKey(argkey);
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
      }
      RAI_Tensor *t = RedisModule_ModuleTypeGetValue(argkey);
      RedisModule_CloseKey(argkey);
      const char* opname = RedisModule_StringPtrLen(argname, NULL);
      if (!RAI_RunCtxAddInput(rinfo->gctx, opname, t)) {
        // todo free rinfo
        return RedisModule_ReplyWithError(ctx, "Input key not found.");
      }
    } else {
      const char* opname = RedisModule_StringPtrLen(argname, NULL);
      if (!RAI_RunCtxAddOutput(rinfo->gctx, opname)) {
        // todo free rinfo
        return RedisModule_ReplyWithError(ctx, "Output key not found.");
      }
      RedisModule_RetainString(ctx, argv[i]);
      rinfo->outkeys[(i-pairoffset)/2-ninputs] = argv[i];
    }
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_Run_Reply, NULL, NULL, 0);

  //  RedisModule_AbortBlock(bc);
  //  return RedisModule_ReplyWithError(ctx, "-ERR Can't start thread");

  pthread_mutex_lock(&runQueueMutex);
  queuePush(runQueue, rinfo);
  pthread_mutex_unlock(&runQueueMutex);

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

int RedisAI_Set_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (argc < 2) return RedisModule_WrongArity(ctx);

  const char* subcommand;
  subcommand = RedisModule_StringPtrLen(argv[1], NULL);
  if (strcasecmp(subcommand, "TENSOR") == 0) {
    return RedisAI_Set_Tensor_RedisCommand(ctx, argv+1, argc-1);
  }
  else if (strcasecmp(subcommand, "GRAPH") == 0) {
    return RedisAI_Set_Graph_RedisCommand(ctx, argv+1, argc-1);
  }
  else if (strcasecmp(subcommand, "SCRIPT") == 0) {
    return RedisModule_ReplyWithError(ctx, "ERR SCRIPT subcommand not implemented yet");
  }

  return RedisModule_ReplyWithError(ctx, "ERR unrecognized subcommand");
}

int RedisAI_Get_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (argc < 2) return RedisModule_WrongArity(ctx);

  const char* subcommand;
  subcommand = RedisModule_StringPtrLen(argv[1], NULL);
  if (strcasecmp(subcommand, "TENSOR") == 0) {
    return RedisAI_Get_Tensor_RedisCommand(ctx, argv+1, argc-1);
  }
  //else if (strcasecmp(subcommand, "GRAPH") == 0) {
  //  return RedisAI_GGet_RedisCommand(ctx, argv+1, argc-1);
  //}
  else if (strcasecmp(subcommand, "SCRIPT") == 0) {
    return RedisModule_ReplyWithError(ctx, "ERR SCRIPT subcommand not implemented yet");
  }

  return RedisModule_ReplyWithError(ctx, "ERR unrecognized subcommand");
}

int RedisAI_Run_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (argc < 2) return RedisModule_WrongArity(ctx);

  const char* subcommand;
  subcommand = RedisModule_StringPtrLen(argv[1], NULL);
  if (strcasecmp(subcommand, "GRAPH") == 0) {
    return RedisAI_Run_Graph_RedisCommand(ctx, argv+1, argc-1);
  }
  else if (strcasecmp(subcommand, "SCRIPT") == 0) {
    return RedisModule_ReplyWithError(ctx, "ERR SCRIPT subcommand not implemented yet");
  }

  return RedisModule_ReplyWithError(ctx, "ERR unrecognized subcommand");
}

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, registerApiCallback) \
  if(registerApiCallback("RedisAI_" #name, RAI_ ## name)){\
      printf("could not register RedisAI_" #name "\r\n");\
      return false;\
  }

static bool RediDL_RegisterApi(int (*registerApiCallback)(const char *funcname, void *funcptr)){
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

  REGISTER_API(GraphCreate, registerApiCallback);
  REGISTER_API(GraphFree, registerApiCallback);
  REGISTER_API(RunCtxCreate, registerApiCallback);
  REGISTER_API(RunCtxAddInput, registerApiCallback);
  REGISTER_API(RunCtxAddOutput, registerApiCallback);
  REGISTER_API(RunCtxNumOutputs, registerApiCallback);
  REGISTER_API(RunCtxOutputTensor, registerApiCallback);
  REGISTER_API(RunCtxFree, registerApiCallback);
  REGISTER_API(GraphRun, registerApiCallback);
  REGISTER_API(GraphGetShallowCopy, registerApiCallback);
  return true;
}

int moduleRegisterApi(const char *funcname, void *funcptr);

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "ai", 1, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  if(!RediDL_RegisterApi(moduleRegisterApi)){
    RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_TensorInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt");
    return REDISMODULE_ERR;
  }

  if(!RAI_GraphInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt");
    return REDISMODULE_ERR;
  }

  if (RedisModule_CreateCommand(ctx, "ai.set", RedisAI_Set_RedisCommand, "write", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.get", RedisAI_Get_RedisCommand, "readonly", 2, 2, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.run", RedisAI_Run_RedisCommand, "write", 2, -1, 2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisAI_StartRunThread() == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}

