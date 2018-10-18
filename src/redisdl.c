#define REDISMODULE_EXPERIMENTAL_API

#include "redismodule.h"
#include "tensorflow/c/c_api.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

static RedisModuleType *RedisDL_TensorType;
static RedisModuleType *RedisDL_GraphType;

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

size_t RedisDL_DataSize(TF_DataType type) {
  switch (type) {
    case TF_BOOL:
      return sizeof(int8_t);
    case TF_INT8:
      return sizeof(int8_t);
    case TF_UINT8:
      return sizeof(uint8_t);
    case TF_INT16:
      return sizeof(int16_t);
    case TF_UINT16:
      return sizeof(uint16_t);
    case TF_INT32:
      return sizeof(int32_t);
    case TF_INT64:
      return sizeof(int64_t);
    case TF_FLOAT:
      return sizeof(float);
    case TF_DOUBLE:
      return sizeof(double);
    case TF_COMPLEX64:
      return 2*sizeof(float);
    case TF_COMPLEX128:
      return 2 * sizeof(double);
    default:
      return 0;
  }
}

int RedisDL_SetValueFromDouble(TF_Tensor* tensor, long long i, double val) {
  switch (TF_TensorType(tensor)) {
    case TF_FLOAT:
      ((float*)TF_TensorData(tensor))[i] = val; break;
    case TF_DOUBLE:
      ((double*)TF_TensorData(tensor))[i] = val; break;
    default:
      return -1;
  }
  return 0;
}

int RedisDL_SetValueFromLongLong(TF_Tensor* tensor, long long i, long long val) {
  switch (TF_TensorType(tensor)) {
    case TF_BOOL:
      ((int8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT8:
      ((int8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_UINT8:
      ((uint8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT16:
      ((int16_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_UINT16:
      ((uint16_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT32:
      ((int32_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT64:
      ((int64_t*)TF_TensorData(tensor))[i] = val; break;
    default:
      return -1;
  }
  return 0;
}

int RedisDL_GetValueAsDouble(TF_Tensor* tensor, long long i, double* val) {
  switch (TF_TensorType(tensor)) {
    case TF_FLOAT:
      *val = ((float*)TF_TensorData(tensor))[i]; break;
    case TF_DOUBLE:
      *val = ((double*)TF_TensorData(tensor))[i]; break;
    default:
      return -1;
  }
  return 0;
}

int RedisDL_GetValueAsLongLong(TF_Tensor* tensor, long long i, long long* val) {
  switch (TF_TensorType(tensor)) {
    case TF_BOOL:
      *val = ((int8_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT8:
      *val = ((int8_t*)TF_TensorData(tensor))[i]; break;
    case TF_UINT8:
      *val = ((uint8_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT16:
      *val = ((int16_t*)TF_TensorData(tensor))[i]; break;
    case TF_UINT16:
      *val = ((uint16_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT32:
      *val = ((int32_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT64:
      *val = ((int64_t*)TF_TensorData(tensor))[i]; break;
    default:
      return -1;
  }
  return 0;
}

TF_Tensor* RedisDL_clone(RedisModuleCtx *ctx, const TF_Tensor *tensor) {
  int ndims = TF_NumDims(tensor);
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (int j=0; j<ndims; j++) {
    dims[j] = TF_Dim(tensor, j);
  }
  size_t len = TF_TensorByteSize(tensor);
  void *data = TF_TensorData(tensor);
  TF_Tensor *out = TF_AllocateTensor(TF_TensorType(tensor),
                                     dims, ndims, len);
  memcpy(TF_TensorData(out), data, len);
  return out;
}

struct RedisDL_TensorTypeObject {
  void *tensor;
};

struct RedisDL_TensorTypeObject *createTensorTypeObject(TF_Tensor* tensor) {
  struct RedisDL_TensorTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->tensor = tensor;
  return o;
}

struct RedisDL_GraphTypeObject {
  void *graph;
  // TODO: use session pool? The ideal would be to use one session per client.
  //       If a client disconnects, we dispose the session or reuse it for
  //       another client.
  void *session;
};

struct RedisDL_GraphTypeObject *createGraphTypeObject(TF_Graph* graph) {
  TF_Status *status = TF_NewStatus();

  TF_SessionOptions *options = TF_NewSessionOptions();
  TF_Session *session = TF_NewSession(graph, options, status);

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return RedisModule_Alloc(sizeof(struct RedisDL_GraphTypeObject));
  }

  TF_DeleteSessionOptions(options);
  TF_DeleteStatus(status);

  struct RedisDL_GraphTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->graph = graph;
  o->session = session;
  return o;
}

void RedisDL_TensorType_ReleaseObject(struct RedisDL_TensorTypeObject* o) {
  TF_DeleteTensor(o->tensor);
  RedisModule_Free(o);
}

void RedisDL_GraphType_ReleaseObject(struct RedisDL_GraphTypeObject* o) {

  TF_Status *status = TF_NewStatus();
  TF_CloseSession(o->session, status);

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteSession(o->session, status);
  o->session = NULL;

  if (TF_GetCode(status) != TF_OK) {
    // TODO: raise error but we don't have a hold on ctx (that's because the caller _Free_ doesn't)
    // return RedisModule_ReplyWithError(ctx, TF_Message(status));
    return;
  }

  TF_DeleteGraph(o->graph);
  o->graph = NULL;

  TF_DeleteStatus(status);

  RedisModule_Free(o);
}

enum RedisDL_DataFmt {
  REDISTF_DATA_BLOB = 0,
  REDISTF_DATA_VALUES
};

// ================================

// key type ndims dim1..dimN [BLOB data | VALUES val1..valN]
int RedisDL_Tensor_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 5) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  size_t typelen;
  TF_DataType datatype;
  char typerr = 0;
  const char* typestr = RedisModule_StringPtrLen(argv[2], &typelen);
  if (strcasecmp(typestr, "FLOAT") == 0) datatype = TF_FLOAT;
  else if (strcasecmp(typestr, "DOUBLE") == 0) datatype = TF_DOUBLE;
  else if (strncasecmp(typestr, "INT", 3) == 0) {
    const char *bitstr = typestr + 3;
    if (strcmp(bitstr, "8") == 0) datatype = TF_INT8;
    else if (strcmp(bitstr, "16") == 0) datatype = TF_INT16;
    else if (strcmp(bitstr, "32") == 0) datatype = TF_INT32;
    else if (strcmp(bitstr, "64") == 0) datatype = TF_INT64;
    else typerr = 1;
  }
  else if (strncasecmp(typestr, "UINT", 4) == 0) {
    const char *bitstr = typestr + 4;
    if (strcmp(bitstr, "8") == 0) datatype = TF_UINT8;
    else if (strcmp(bitstr, "16") == 0) datatype = TF_UINT16;
    else typerr = 1;
  }
  else typerr = 1;

  if (typerr) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR invalid type");
  }

  long long ndims = 0;
  if ((RedisModule_StringToLongLong(argv[3], &ndims) != REDISMODULE_OK)
      || ndims < 0) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR invalid ndims");
  }

  if (argc < 4 + ndims) {
    RedisModule_CloseKey(key);
    return RedisModule_WrongArity(ctx);
  }

  long long len = 1;
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (long long i=0; i<ndims; i++) {
    if ((RedisModule_StringToLongLong(argv[4+i], dims+i) != REDISMODULE_OK)
        || dims[i] < 0) {
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "ERR invalid dims");
    }
    len *= dims[i];
  }

  if (argc != 4 + ndims &&
      argc != 4 + ndims + 1 + 1 &&
      argc != 4 + ndims + 1 + len) {
    RedisModule_CloseKey(key);
    return RedisModule_WrongArity(ctx);
  }

  int datafmt_arg = 4 + ndims;
  int hasdata = argc > datafmt_arg;

  size_t fmtlen;
  const char* fmtstr;
  int datafmt;
  if (hasdata) {
    fmtstr = RedisModule_StringPtrLen(argv[datafmt_arg], &fmtlen);
    if (strcasecmp(fmtstr, "BLOB") == 0) datafmt = REDISTF_DATA_BLOB;
    else if (strcasecmp(fmtstr, "VALUES") == 0) datafmt = REDISTF_DATA_VALUES;
    else {
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "ERR unsupported data format");
    }
  }

  size_t datasize = RedisDL_DataSize(datatype);
  if (datasize == 0) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR unsupported type");
  }
  size_t nbytes = len * datasize;

  size_t datalen;
  const char* data;
  if (hasdata) {
    data = RedisModule_StringPtrLen(argv[datafmt_arg + 1], &datalen);
    if (datafmt == REDISTF_DATA_BLOB && datalen != nbytes) {
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "ERR data length does not match tensor shape and type");
    }
  }

  if (hasdata && datafmt == REDISTF_DATA_VALUES && argc != len + 5 + ndims) {
    RedisModule_CloseKey(key);
    return RedisModule_WrongArity(ctx);
  }

  TF_Tensor* tensor = TF_AllocateTensor((TF_DataType)datatype, dims, ndims, nbytes);

  if (hasdata && datafmt == REDISTF_DATA_BLOB) {
    memcpy(TF_TensorData(tensor), data, datalen);
  }
  else if (hasdata && datafmt == REDISTF_DATA_VALUES) {
    long i;
    if (datatype == TF_FLOAT || datatype == TF_DOUBLE) {
      double val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToDouble(argv[datafmt_arg + 1 + i], &val) != REDISMODULE_OK)) {
          TF_DeleteTensor(tensor);
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR invalid val");
        }
        int ret = RedisDL_SetValueFromDouble(tensor, i, val);
        if (ret == -1) {
          TF_DeleteTensor(tensor);
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
    else {
      long long val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToLongLong(argv[datafmt_arg + 1 + i], &val) != REDISMODULE_OK)) {
          TF_DeleteTensor(tensor);
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR invalid val");
        }
        int ret = RedisDL_SetValueFromLongLong(tensor, i, val);
        if (ret == -1) {
          TF_DeleteTensor(tensor);
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
        }
      }
    }
  }

  struct RedisDL_TensorTypeObject *tto = createTensorTypeObject(tensor);
  RedisModule_ModuleTypeSetValue(key, RedisDL_TensorType, tto);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

int RedisDL_Type_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  TF_DataType data_type = TF_TensorType(tto->tensor);

  RedisModule_CloseKey(key);

  switch (data_type) {
  case TF_FLOAT:
    return RedisModule_ReplyWithSimpleString(ctx, "FLOAT");
  case TF_DOUBLE:
    return RedisModule_ReplyWithSimpleString(ctx, "DOUBLE");
  case TF_INT8:
    return RedisModule_ReplyWithSimpleString(ctx, "INT8");
  case TF_INT16:
    return RedisModule_ReplyWithSimpleString(ctx, "INT16");
  case TF_INT32:
    return RedisModule_ReplyWithSimpleString(ctx, "INT32");
  case TF_INT64:
    return RedisModule_ReplyWithSimpleString(ctx, "INT64");
  case TF_UINT8:
    return RedisModule_ReplyWithSimpleString(ctx, "UINT8");
  case TF_UINT16:
    return RedisModule_ReplyWithSimpleString(ctx, "UINT16");
  default:
    return RedisModule_ReplyWithError(ctx, "ERR unsupported type");
  }
}

int RedisDL_NDims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  RedisModule_CloseKey(key);

  return RedisModule_ReplyWithLongLong(ctx, ndims);
}

int RedisDL_Dims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  RedisModule_ReplyWithArray(ctx, ndims);
  for (long long i=0; i<ndims; i++) {
    long long dim = TF_Dim(tto->tensor, i);
    RedisModule_ReplyWithLongLong(ctx, dim);
  }

  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

int RedisDL_Size_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);

  RedisModule_CloseKey(key);

  return RedisModule_ReplyWithLongLong(ctx, size);
}

int RedisDL_Data_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);
  char *data = TF_TensorData(tto->tensor);

  int ret = RedisModule_ReplyWithStringBuffer(ctx, data, size);

  RedisModule_CloseKey(key);

  return ret;
}

int RedisDL_Values_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_TensorType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);
  long long len = 1;
  long long i;
  for (i=0; i<ndims; i++) {
    len *= TF_Dim(tto->tensor, i);
  }

  TF_DataType datatype = TF_TensorType(tto->tensor);

  RedisModule_ReplyWithArray(ctx, len);

  if (datatype == TF_FLOAT || datatype == TF_DOUBLE) {
    double val;
    for (i=0; i<len; i++) {
      int ret = RedisDL_GetValueAsDouble(tto->tensor, i, &val);
      if (ret == -1) {
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
      }
      RedisModule_ReplyWithDouble(ctx, val);
    }
  }
  else {
    long long val;
    for (i=0; i<len; i++) {
      int ret = RedisDL_GetValueAsLongLong(tto->tensor, i, &val);
      if (ret == -1) {
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, "ERR cannot get values for this datatype");
      }
      RedisModule_ReplyWithLongLong(ctx, val);
    }
  }

  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}


// ================================

// key graphbuf [prefix]
int RedisDL_Graph_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if ((argc != 3) && (argc != 4)) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != RedisDL_GraphType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  size_t graphlen;
  const char* graphdef = RedisModule_StringPtrLen(argv[2], &graphlen);

  TF_Graph* graph = TF_NewGraph();

  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();

  if (argc == 4) {
    size_t prefixlen;
    const char* prefix = RedisModule_StringPtrLen(argv[3], &prefixlen);
    TF_ImportGraphDefOptionsSetPrefix(options, prefix);
  }
  else {
    TF_ImportGraphDefOptionsSetPrefix(options, "");
  }

  TF_Buffer *buffer = TF_NewBuffer();
  buffer->length = graphlen;
  buffer->data = graphdef;

  TF_Status *status = TF_NewStatus();

  TF_GraphImportGraphDef(graph, buffer, options, status);

  if (TF_GetCode(status) != TF_OK) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, TF_Message(status));
  }

  TF_DeleteImportGraphDefOptions(options);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(status);

  struct RedisDL_GraphTypeObject *gto = createGraphTypeObject(graph);
  RedisModule_ModuleTypeSetValue(key, RedisDL_GraphType, gto);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

struct RedisDL_RunInfo {
  RedisModuleBlockedClient *client;
  TF_Session *session;
  TF_Output *inputs;
  TF_Tensor **input_values;
  long long ninputs;
  TF_Output *outputs;
  TF_Tensor **output_values;
  long long noutputs;
  RedisModuleKey *graphkey;
  RedisModuleString **outkeys;
  TF_Status *status;
};

void RedisDL_FreeRunInfo(struct RedisDL_RunInfo *rinfo) {
  RedisModule_Free(rinfo->inputs);
  for (int i=0; i<rinfo->ninputs; i++) {
    TF_DeleteTensor(rinfo->input_values[i]);
  }
  RedisModule_Free(rinfo->input_values);
  RedisModule_Free(rinfo->outputs);
  RedisModule_Free(rinfo->output_values);
  RedisModule_Free(rinfo->outkeys);
  TF_DeleteStatus(rinfo->status);
  RedisModule_Free(rinfo);
}

void *RedisDL_RunSession(void *arg) {
  struct RedisDL_RunInfo *rinfo = (struct RedisDL_RunInfo*)arg;

  RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(rinfo->client);

  mstime_t start = mstime();

  TF_Status *status = TF_NewStatus();

  TF_SessionRun(rinfo->session, NULL /* run_options */,
                rinfo->inputs, rinfo->input_values, rinfo->ninputs,
                rinfo->outputs, rinfo->output_values, rinfo->noutputs,
                NULL /* target_opers */, 0 /* ntargets */,
                NULL /* run_Metadata */,
                rinfo->status);

  mstime_t end = mstime();

  RedisModule_ThreadSafeContextLock(ctx);
  RedisModule_Log(ctx, "notice", "TF_SessionRun took %fs", (end - start) / 1000.0);
  RedisModule_ThreadSafeContextUnlock(ctx);

  RedisModule_FreeThreadSafeContext(ctx);
  RedisModule_UnblockClient(rinfo->client, rinfo);

  return NULL;
}

//void RedisDL_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
//  RedisModule_Log(ctx,"warning","Blocked client %p disconnected!", (void*)bc);
//
//  // TODO: clean up
//}

int RedisDL_Run_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  struct RedisDL_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);

  if (TF_GetCode(rinfo->status) != TF_OK) {
    int ret = RedisModule_ReplyWithError(ctx, TF_Message(rinfo->status));
    RedisDL_FreeRunInfo(rinfo);
    return ret;
  }

  for (int i=0; i<rinfo->noutputs; i++) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx, rinfo->outkeys[i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        RedisModule_ModuleTypeGetType(outkey) != RedisDL_TensorType) {
      RedisModule_CloseKey(outkey);
      RedisDL_FreeRunInfo(rinfo);
      return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    struct RedisDL_TensorTypeObject *tto = createTensorTypeObject(rinfo->output_values[i]);
    RedisModule_ModuleTypeSetValue(outkey, RedisDL_TensorType, tto);
    RedisModule_CloseKey(outkey);
  }

  // FIXME This crashes Redis, we need to investigate.
  //RedisModule_CloseKey(rinfo->graphkey);

  RedisDL_FreeRunInfo(rinfo);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

// graph key, ninputs, (input key, input name)..., (output key, output name)...
int RedisDL_Run_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
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
  if (RedisModule_ModuleTypeGetType(key) != RedisDL_GraphType) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisDL_GraphTypeObject *gto = RedisModule_ModuleTypeGetValue(key);

  TF_Graph* graph = gto->graph;
  TF_Session* session = gto->session;

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

  TF_Output *inputs = RedisModule_Alloc(ninputs*sizeof(TF_Output));
  TF_Output *outputs = RedisModule_Alloc(noutputs*sizeof(TF_Output));
  TF_Tensor **input_values = RedisModule_Alloc(ninputs*sizeof(TF_Tensor*));
  TF_Tensor **output_values = RedisModule_Alloc(noutputs*sizeof(TF_Tensor*));

  RedisModuleString **outkeys = RedisModule_Alloc(noutputs*sizeof(RedisModuleString*));

  for (int i=pairoffset; i<argc; i+=2) {
    int isinput = i < pairoffset + 2 * ninputs;

    size_t namelen;
    RedisModuleString* argname = argv[i+1];

    if (isinput) {
      RedisModuleKey *argkey = RedisModule_OpenKey(ctx, argv[i], REDISMODULE_READ);
      if (RedisModule_ModuleTypeGetType(argkey) != RedisDL_TensorType) {
        RedisModule_CloseKey(argkey);
        return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
      }
      struct RedisDL_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(argkey);
      input_values[(i-pairoffset)/2] = RedisDL_clone(ctx, tto->tensor);
      RedisModule_CloseKey(argkey);
      const char* opname = RedisModule_StringPtrLen(argname, &namelen);
      // RedisModule_Log(ctx, "warning", "%s", opname);
      TF_Output port;
      port.oper = TF_GraphOperationByName(graph, opname);
      port.index = 0;
      if (port.oper == NULL) {
        return RedisModule_ReplyWithError(ctx, "Input key not found.");
      }
      inputs[(i-pairoffset)/2] = port;
    } else {
      const char* opname = RedisModule_StringPtrLen(argname, &namelen);
      TF_Output port;
      port.oper = TF_GraphOperationByName(graph, opname);
      port.index = 0;
      if (port.oper == NULL) {
        return RedisModule_ReplyWithError(ctx, "Output key not found.");
      }
      outputs[(i-pairoffset)/2-ninputs] = port;
      outkeys[(i-pairoffset)/2-ninputs] = argv[i];
    }
  }

  RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, RedisDL_Run_Reply, NULL, NULL, 0);

  struct RedisDL_RunInfo *rinfo = RedisModule_Alloc(sizeof(struct RedisDL_RunInfo));
  rinfo->client = bc;
  rinfo->session = session;
  rinfo->inputs = inputs;
  rinfo->input_values = input_values;
  rinfo->ninputs = ninputs;
  rinfo->outputs = outputs;
  rinfo->output_values = output_values;
  rinfo->noutputs = noutputs;
  rinfo->graphkey = key;
  rinfo->outkeys = outkeys;
  rinfo->status = TF_NewStatus();

  //  RedisModule_AbortBlock(bc);
  //  return RedisModule_ReplyWithError(ctx, "-ERR Can't start thread");

  pthread_mutex_lock(&runQueueMutex);
  queuePush(runQueue, rinfo);
  pthread_mutex_unlock(&runQueueMutex);

  return REDISMODULE_OK;
}

// ================================

void *RedisDL_TensorType_RdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning", "Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void RedisDL_TensorType_RdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void RedisDL_TensorType_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

unsigned long RedisDL_TensorType_MemUsage(const void *value) {
  // TODO
  return 0;
}

void RedisDL_TensorType_Digest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void RedisDL_TensorType_Free(void *value) {
  RedisDL_TensorType_ReleaseObject(value);
}

// ================================

void *RedisDL_GraphType_RdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning", "Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void RedisDL_GraphType_RdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void RedisDL_GraphType_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

unsigned long RedisDL_GraphType_MemUsage(const void *value) {
  // TODO
  return 0;
}

void RedisDL_GraphType_Digest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void RedisDL_GraphType_Free(void *value) {
  RedisDL_GraphType_ReleaseObject(value);
}

void *RedisDL_Run_ThreadMain(void *arg) {

  while(1) {
    pthread_mutex_lock(&runQueueMutex);
    queueItem *item = queuePop(runQueue);
    pthread_mutex_unlock(&runQueueMutex);

    if (item) {
      RedisDL_RunSession(item->value);
    }

    // release item (note: the callback does it; verify)
    // unblock (the callback does it)

    usleep(1000);
  }
}

int RedisDL_StartRunThread() {
  pthread_t tid;

  runQueue = queueCreate();

  if (pthread_create(&tid, NULL, RedisDL_Run_ThreadMain, NULL) != 0) {
    return REDISMODULE_ERR;
  }

  return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "dl", 1, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

   RedisModuleTypeMethods tmTensor = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RedisDL_TensorType_RdbLoad,
      .rdb_save = RedisDL_TensorType_RdbSave,
      .aof_rewrite = RedisDL_TensorType_AofRewrite,
      .mem_usage = RedisDL_TensorType_MemUsage,
      .free = RedisDL_TensorType_Free,
      .digest = RedisDL_TensorType_Digest
  };

  RedisDL_TensorType = RedisModule_CreateDataType(ctx, "TF_TENSOR", 0, &tmTensor);
  if (RedisDL_TensorType == NULL) return REDISMODULE_ERR;

   RedisModuleTypeMethods tmGraph = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RedisDL_GraphType_RdbLoad,
      .rdb_save = RedisDL_GraphType_RdbSave,
      .aof_rewrite = RedisDL_GraphType_AofRewrite,
      .mem_usage = RedisDL_GraphType_MemUsage,
      .free = RedisDL_GraphType_Free,
      .digest = RedisDL_GraphType_Digest
  };

  RedisDL_GraphType = RedisModule_CreateDataType(ctx, "TF_TENSOR", 0, &tmGraph);
  if (RedisDL_GraphType == NULL) return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.tensor", RedisDL_Tensor_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.type", RedisDL_Type_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.ndims", RedisDL_NDims_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.dims", RedisDL_Dims_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.size", RedisDL_Size_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.data", RedisDL_Data_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.values", RedisDL_Values_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.graph", RedisDL_Graph_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "dl.run", RedisDL_Run_RedisCommand, "write", 1, -1, 2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisDL_StartRunThread() == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}

