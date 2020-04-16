#include "redismodule.h"
#include "tensor.h"
#include "model.h"
#include "script.h"
#include "backends.h"
#include "stats.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"
#include "rmutil/args.h"

#define REDISAI_H_INCLUDE
#include "redisai.h"
#undef REDISAI_H_INCLUDE

typedef struct RunQueueInfo {
  pthread_mutex_t run_queue_mutex;
  pthread_cond_t queue_condition_var;
  queue *run_queue;
  pthread_t *threads;
} RunQueueInfo;

static AI_dict *run_queues = NULL;
static long long perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;

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

/* Return REDISMODULE_ERR if there was an error getting the Model.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key, RAI_Model **model,
                              int mode) {
  *key = RedisModule_OpenKey(ctx, keyName, mode);
  if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, "ERR model key is empty");
    return REDISMODULE_ERR;
  }
  if (RedisModule_ModuleTypeGetType(*key) != RedisAI_ModelType) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    return REDISMODULE_ERR;
  }
  *model = RedisModule_ModuleTypeGetValue(*key);
  return REDISMODULE_OK;
}

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key, RAI_Script **script,
                              int mode) {
  *key = RedisModule_OpenKey(ctx, keyName, mode);
  if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, "ERR script key is empty");
    return REDISMODULE_ERR;
  }
  if (RedisModule_ModuleTypeGetType(*key) != RedisAI_ScriptType) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    return REDISMODULE_ERR;
  }
  *script = RedisModule_ModuleTypeGetValue(*key);
  return REDISMODULE_OK;
}

/* Return REDISMODULE_ERR if is the key not associated with a tensor type.
 * Return REDISMODULE_OK otherwise. */
int RAI_OpenKey_Tensor(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key,
                              int mode) {
  *key = RedisModule_OpenKey(ctx, keyName, mode);
  if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) { 
    return REDISMODULE_OK;
  }
  if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    return REDISMODULE_ERR;
  }
  return REDISMODULE_OK;
}

/* Return REDISMODULE_ERR if there was an error getting the Tensor.
 * Return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable. */
int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                               RedisModuleKey **key, RAI_Tensor **tensor,
                               int mode) {
  *key = RedisModule_OpenKey(ctx, keyName, mode);
  if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, "ERR tensor key is empty");
    return REDISMODULE_ERR;
  }
  if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
    RedisModule_CloseKey(*key);
    RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
    return REDISMODULE_ERR;
  }
  *tensor = RedisModule_ModuleTypeGetValue(*key);
  return REDISMODULE_OK;
}

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

void RedisAI_FreeRunStats(RedisModuleCtx *ctx, struct RedisAI_RunStats *rstats) {
  RedisModule_FreeString(ctx, rstats->key);
  RedisModule_Free(rstats->devicestr);
}

void *RedisAI_RunSession(struct RedisAI_RunInfo **batch_rinfo) {

  const long long batch_size = array_len(batch_rinfo);

  if (batch_size == 0) {
    return NULL;
  }

  RAI_ModelRunCtx** mctxs = NULL;
  RAI_ScriptRunCtx* sctx = NULL;

  RAI_Error* err = RedisModule_Calloc(1, sizeof(RAI_Error));
  long long rtime;
  int status;
  if (batch_rinfo[0]->mctx) {
    mctxs = array_new(RAI_ModelRunCtx*, batch_size);
    for (long long i=0; i<batch_size; i++) {
      mctxs = array_append(mctxs, batch_rinfo[i]->mctx);
    }
  }
  else if (batch_rinfo[0]->sctx) {
    sctx = batch_rinfo[0]->sctx;
  }

  const long long start = ustime();
  if (mctxs) {
    status = RAI_ModelRun(mctxs, err);
  }
  else if (sctx) {
    status = RAI_ScriptRun(sctx, err);
  }
  rtime = ustime() - start;

  for (long long i=0; i<batch_size; i++) {
    struct RedisAI_RunInfo *rinfo = batch_rinfo[i];

    rinfo->status = status;
    rinfo->err = RedisModule_Calloc(1, sizeof(RAI_Error));
    // TODO: add information on whether the call was batched
    // and how large the batch was
    rinfo->duration_us = rtime;

    rinfo->err->code = err->code;
    if (err->code != RAI_OK) {
      rinfo->err->detail = RedisModule_Strdup(err->detail);
      rinfo->err->detail_oneline = RedisModule_Strdup(err->detail_oneline);
    }
    if (rinfo->client != NULL) {
      RedisModule_UnblockClient(rinfo->client, rinfo);
    }
  }

  if (mctxs) {
    array_free(mctxs);
  }
  else if (sctx) {
    // No batching for scripts for now
  }

  return NULL;
}

void RedisAI_FreeData(RedisModuleCtx *ctx, void *rinfo) {
}

void RedisAI_Disconnected(RedisModuleCtx *ctx, RedisModuleBlockedClient *bc) {
  RedisModule_Log(ctx, "warning", "Blocked client %p disconnected!", (void*)bc);
}

void RedisAI_ReplicateTensorSet(RedisModuleCtx *ctx, RedisModuleString *key, RAI_Tensor *t) {
  long long ndims = RAI_TensorNumDims(t);

  char *dtypestr = NULL;
  Tensor_DataTypeStr(RAI_TensorDataType(t), &dtypestr);

  assert(dtypestr);

  char *data = RAI_TensorData(t);
  long long size = RAI_TensorByteSize(t);

  RedisModuleString* dims[ndims];

  for (long long i=0; i<ndims; i++) {
    dims[i] = RedisModule_CreateStringFromLongLong(ctx, RAI_TensorDim(t, i));
  }

  RedisModule_Replicate(ctx, "AI.TENSORSET", "scvcb", key, dtypestr,
                        dims, ndims, "BLOB", data, size);

  for (long long i=0; i<ndims; i++) {
    RedisModule_FreeString(ctx,dims[i]);
  }

  RedisModule_Free(dtypestr);
}

int RedisAI_Run_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  struct RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);
  
  const char* runkey = RedisModule_StringPtrLen(rinfo->runkey, NULL);
  AI_dictEntry *stats_entry = AI_dictFind(run_stats, runkey);

  struct RedisAI_RunStats *rstats = NULL;
  if (stats_entry) {
    rstats = AI_dictGetVal(stats_entry);
  }

  if (rinfo->status) {
    RedisModule_Log(ctx, "warning", "ERR %s", rinfo->err->detail);
    if (rstats) {
      rstats->calls += 1;
      rstats->nerrors += 1;
    }
    int ret = RedisModule_ReplyWithError(ctx, rinfo->err->detail_oneline);
    RedisAI_FreeRunInfo(ctx, rinfo);
    return ret;
  }

  size_t num_outputs = 0;
  if (rinfo->mctx) {
    num_outputs = RAI_ModelRunCtxNumOutputs(rinfo->mctx);
  }
  else if (rinfo->sctx) {
    num_outputs = RAI_ScriptRunCtxNumOutputs(rinfo->sctx);
  }

  int64_t batch_size = 0;

  for (size_t i=0; i<num_outputs; ++i) {
    RedisModuleKey *outkey;
    const int status = RAI_OpenKey_Tensor(ctx, rinfo->outkeys[i], &outkey, REDISMODULE_READ|REDISMODULE_WRITE);
    if(status==REDISMODULE_ERR){
        RedisAI_FreeRunInfo(ctx, rinfo);
        if (rstats) {
          rstats->calls += 1;
          rstats->nerrors += 1;
        }
        return REDISMODULE_ERR;
    }
    RAI_Tensor *t = NULL;
    if (rinfo->mctx) {
      t = RAI_ModelRunCtxOutputTensor(rinfo->mctx, i);
      if (t && batch_size == 0) {
        batch_size = RAI_TensorDim(t, 0);
      }
    }
    else if (rinfo->sctx) {
      t = RAI_ScriptRunCtxOutputTensor(rinfo->sctx, i);
    }
    if (t) {
      RedisModule_ModuleTypeSetValue(outkey, RedisAI_TensorType, RAI_TensorGetShallowCopy(t));
    }
    RedisModule_CloseKey(outkey);

    if (t) {
      RedisAI_ReplicateTensorSet(ctx, rinfo->outkeys[i], t);
    }
  }

  if (rstats) {
    rstats->duration_us += rinfo->duration_us;
    rstats->calls += 1;

    if (rinfo->mctx) {
      rstats->samples += batch_size;
    }
  }

  // FIXME This crashes Redis, we need to investigate.
  //RedisModule_CloseKey(rinfo->modelkey);

  RedisAI_FreeRunInfo(ctx, rinfo);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

size_t RAI_RunInfoBatchSize(struct RedisAI_RunInfo* rinfo) {
  if (rinfo->mctx == NULL) {
    return -1;
  }

  size_t ninputs = RAI_ModelRunCtxNumInputs(rinfo->mctx);

  int batchsize = 0;

  if (ninputs == 0) {
    return batchsize;
  }

  for (size_t i=0; i<ninputs; i++) {
    RAI_Tensor* input = RAI_ModelRunCtxInputTensor(rinfo->mctx, i);

    if (i == 0) {
      batchsize = RAI_TensorDim(input, 0);
      continue;
    }

    if (batchsize != RAI_TensorDim(input, 0)) {
      batchsize = 0;
      break;
    }
  }

  return batchsize;
}

int RAI_RunInfoBatchable(struct RedisAI_RunInfo* rinfo1, struct RedisAI_RunInfo* rinfo2) {
  if (rinfo1->mctx == NULL || rinfo2->mctx == NULL) {
    return 0;
  }

  if (rinfo1->mctx->model != rinfo2->mctx->model) {
    return 0;
  }

  int ninputs1 = RAI_ModelRunCtxNumInputs(rinfo1->mctx);
  int ninputs2 = RAI_ModelRunCtxNumInputs(rinfo2->mctx);

  if (ninputs1 != ninputs2) {
    return 0;
  }

  for (int i=0; i<ninputs1; i++) {
    RAI_Tensor* input1 = RAI_ModelRunCtxInputTensor(rinfo1->mctx, i);
    RAI_Tensor* input2 = RAI_ModelRunCtxInputTensor(rinfo2->mctx, i);

    int ndims1 = RAI_TensorNumDims(input1);
    int ndims2 = RAI_TensorNumDims(input2);

    if (ndims1 != ndims2) {
      return 0;
    }

    if (ndims1 == 0) {
      continue;
    }

    for (int j=1; j<ndims1; j++) {
      int dim1 = RAI_TensorDim(input1, j);
      int dim2 = RAI_TensorDim(input2, j);
      if (dim1 != dim2) {
        return 0;
      }
    }
  }

  return 1;
}

void *RedisAI_Run_ThreadMain(void *arg) {

  RunQueueInfo* run_queue_info = (RunQueueInfo*)arg;

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  while (true){
    int rc = pthread_cond_wait(&run_queue_info->queue_condition_var, &run_queue_info->run_queue_mutex);

    long long run_queue_len = queueLength(run_queue_info->run_queue);

    while (run_queue_len > 0) {
      queueItem **evicted_items = NULL;
      struct RedisAI_RunInfo **batch_rinfo = NULL;

      queueItem *item = queueFront(run_queue_info->run_queue);

      while (item) {
        struct RedisAI_RunInfo *rinfo = (struct RedisAI_RunInfo *)item->value;

        if (evicted_items) {
          array_free(evicted_items);
          array_free(batch_rinfo);
        }
        evicted_items = array_new(queueItem *, run_queue_len);
        batch_rinfo = array_new(struct RedisAI_RunInfo *, run_queue_len);

        evicted_items = array_append(evicted_items, item);
        batch_rinfo = array_append(batch_rinfo, rinfo);

        if (rinfo->sctx) {
          break;
        }

        size_t batchsize = rinfo->mctx->model->opts.batchsize;

        if (batchsize == 0) {
          break;
        }

        size_t current_batchsize = RAI_RunInfoBatchSize(rinfo);

        if (current_batchsize == 0 ||
            current_batchsize >= batchsize) {
          break;
        }

        queueItem *next_item = item->next;

        while (next_item != NULL) {
          struct RedisAI_RunInfo *next_rinfo = (struct RedisAI_RunInfo *)next_item->value;

          if (RAI_RunInfoBatchable(rinfo, next_rinfo) == 0) {
            next_item = queueNext(next_item);
            continue;
          }

          int next_batchsize = RAI_RunInfoBatchSize(next_rinfo);

          if (current_batchsize + next_batchsize > batchsize) {
            break;
          }

          evicted_items = array_append(evicted_items, next_item);
          batch_rinfo = array_append(batch_rinfo, next_rinfo);

          current_batchsize += next_batchsize;
          next_item = queueNext(next_item);
        }

        size_t minbatchsize = rinfo->mctx->model->opts.minbatchsize;

        if (minbatchsize == 0 || current_batchsize >= minbatchsize) {
          break;
        }

        item = item->next;
      }

      if (item == NULL) {
        array_free(evicted_items);
        array_free(batch_rinfo);
        pthread_mutex_unlock(&run_queue_info->run_queue_mutex);
        break;
      }

      for (long long i=0; i<array_len(evicted_items); i++) {
        queueEvict(run_queue_info->run_queue, evicted_items[i]);
      }

      pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

      RedisAI_RunSession(batch_rinfo);

      for (long long i=0; i<array_len(evicted_items); i++) {
        RedisModule_Free(evicted_items[i]);
      }
      array_free(evicted_items);
      array_free(batch_rinfo);

      pthread_mutex_lock(&run_queue_info->run_queue_mutex);

      run_queue_len = queueLength(run_queue_info->run_queue);
    }
  }
}

/* ----------------------- RedisAI Module Commands ------------------------- */

/**
 * AI.TENSORSET key type dim1..dimN [BLOB data | VALUES val1..valN]
 */
int RedisAI_TensorSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key;
  const int status = RAI_OpenKey_Tensor(ctx, argv[1], &key, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  // get the tensor datatype
  const char* typestr = RedisModule_StringPtrLen(argv[2], NULL);
  size_t datasize = RAI_TensorDataSizeFromString(typestr);
  if (!datasize){
    return RedisModule_ReplyWithError(ctx, "ERR invalid data type");
  }

  const char* fmtstr;
  int datafmt = REDISAI_DATA_NONE;
  int tensorAllocMode = TENSORALLOC_CALLOC;
  size_t ndims = 0;
  long long len = 1;
  long long* dims = NULL;
  size_t argpos = 3;
  long long remaining_args = argc-1;

  for (; argpos <= argc-1; argpos++){
    const char *opt = RedisModule_StringPtrLen(argv[argpos], NULL);
    remaining_args = argc-1-argpos;
    if (!strcasecmp(opt, "BLOB")){
      datafmt = REDISAI_DATA_BLOB;
      tensorAllocMode = TENSORALLOC_NONE;
      // if we've found the dataformat there are no more dimensions
      // check right away if the arity is correct
      if (remaining_args != 1 ){
        RedisModule_Free(dims);
        RedisModule_CloseKey(key);
        return RedisModule_WrongArity(ctx);
      }
      argpos++;
      break;
    }
    else if (!strcasecmp(opt, "VALUES")){
      datafmt = REDISAI_DATA_VALUES;
      tensorAllocMode = TENSORALLOC_ALLOC;
      //if we've found the dataformat there are no more dimensions
      // check right away if the arity is correct
      if (remaining_args != len ){ 
        RedisModule_Free(dims);
        RedisModule_CloseKey(key);
        return RedisModule_WrongArity(ctx);
      }
      argpos++;
      break;
    } else {
      long long dimension = 1;
      const int retval = RedisModule_StringToLongLong(argv[argpos],&dimension);
      if (retval != REDISMODULE_OK || dimension <= 0) {
          RedisModule_Free(dims);
          RedisModule_CloseKey(key);
          return RedisModule_ReplyWithError(ctx,
              "ERR invalid or negative value found in tensor shape");
      }
      
      ndims++;
      dims=RedisModule_Realloc(dims,ndims*sizeof(long long));
      dims[ndims-1]=dimension;
      len *= dimension;
    }
  }

  const long long nbytes = len * datasize;
  size_t datalen;
  const char *data;
  DLDataType datatype = RAI_TensorDataTypeFromString(typestr);
  RAI_Tensor *t = RAI_TensorCreateWithDLDataType(datatype, dims, ndims, tensorAllocMode);
  if (!t){
    RedisModule_Free(dims);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR could not create tensor");
  }
  size_t i = 0;
  switch (datafmt){
    case REDISAI_DATA_BLOB:
      RedisModule_StringPtrLen(argv[argpos],&datalen);
      if (datalen != nbytes){
        RAI_TensorFree(t);
        RedisModule_CloseKey(key);
        return RedisModule_ReplyWithError(ctx, "ERR data length does not match tensor shape and type");
      }
      RedisModule_RetainString(NULL,argv[argpos]);
      RAI_TensorSetDataFromRS(t,argv[argpos]);
      break;
    case REDISAI_DATA_VALUES:
      for (; argpos <= argc-1; argpos++){
        if (datatype.code == kDLFloat){
          double val;
          const int retval = RedisModule_StringToDouble(argv[argpos],&val);
          if (retval != REDISMODULE_OK) {
            RAI_TensorFree(t);
            RedisModule_CloseKey(key);
            return RedisModule_ReplyWithError(ctx, "ERR invalid value");
          }
          const int retset = RAI_TensorSetValueFromDouble(t, i, val);
          if (retset == -1){
            RAI_TensorFree(t);
            RedisModule_CloseKey(key);
            return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
          }
        }
        else{
          long long val;
          const int retval = RedisModule_StringToLongLong(argv[argpos],&val);
          if (retval != REDISMODULE_OK) {
            RAI_TensorFree(t);
            RedisModule_CloseKey(key);
            return RedisModule_ReplyWithError(ctx, "ERR invalid value");
          }
          const int retset = RAI_TensorSetValueFromLongLong(t, i, val);
          if (retset == -1){
            RAI_TensorFree(t);
            RedisModule_CloseKey(key);
            return RedisModule_ReplyWithError(ctx, "ERR cannot specify values for this datatype");
          }
        }
        i++;
      }
      break;
    default:
      // default does not require tensor data setting since calloc setted it to 0
      break;
  }
  if( RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t) != REDISMODULE_OK ){
    RAI_TensorFree(t);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
  }
  RedisModule_CloseKey(key);
  RedisModule_ReplyWithSimpleString(ctx, "OK");
  RedisModule_ReplicateVerbatim(ctx);
  return REDISMODULE_OK;
}

/**
* AI.TENSORGET tensor_key [BLOB | VALUES | META]
*/
int RedisAI_TensorGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 3) return RedisModule_WrongArity(ctx);

  RAI_Tensor *t;
  RedisModuleKey *key;
  const int status = RAI_GetTensorFromKeyspace(ctx, argv[1], &key, &t, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  int datafmt;
  long long resplen = 2;
  const char *fmtstr = RedisModule_StringPtrLen(argv[2], NULL);
  if (!strcasecmp(fmtstr, "BLOB")) {
    datafmt = REDISAI_DATA_BLOB;
    resplen = 3;
  } else if (!strcasecmp(fmtstr, "VALUES")) {
    datafmt = REDISAI_DATA_VALUES;
    resplen = 3;
  } else if (!strcasecmp(fmtstr, "META")) {
    datafmt = REDISAI_DATA_NONE;
  } else {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR unsupported data format");
  }

  const long long ndims = RAI_TensorNumDims(t);
  
  RedisModule_ReplyWithArray(ctx, resplen);

  char *dtypestr = NULL;
  const int dtypestr_result = Tensor_DataTypeStr(RAI_TensorDataType(t), &dtypestr);
  if(dtypestr_result==REDISMODULE_ERR){
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR unsupported dtype");
  }
  RedisModule_ReplyWithSimpleString(ctx, dtypestr);

  RedisModule_ReplyWithArray(ctx, ndims);
  for (long long i=0; i<ndims; i++) {
    const long long dim = RAI_TensorDim(t, i);
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

/**
* AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]] [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
*/
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
  else if (strcasecmp(bckstr, "TFLITE") == 0) {
    backend = RAI_BACKEND_TFLITE;
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

  if (strlen(devicestr) > 10) {
    return RedisModule_ReplyWithError(ctx, "ERR Invalid DEVICE");
  }

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  unsigned long long batchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "BATCHSIZE")) {
    if (backend == RAI_BACKEND_TFLITE) {
      return RedisModule_ReplyWithError(ctx, "ERR Auto-batching not supported by the TFLITE backend");
    }
    if (AC_GetUnsignedLongLong(&ac, &batchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for BATCHSIZE");
    }
  }

  unsigned long long minbatchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "MINBATCHSIZE")) {
    if (batchsize == 0) {
      return RedisModule_ReplyWithError(ctx, "ERR MINBATCHSIZE specified without BATCHSIZE");
    }
    if (AC_GetUnsignedLongLong(&ac, &minbatchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHSIZE");
    }
  }


  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
  }

  ArgsCursor optionsac;
  AC_GetSliceToOffset(&ac, &optionsac, argc-2);

  if (optionsac.argc == 0 && backend == RAI_BACKEND_TENSORFLOW) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, INPUTS and OUTPUTS not specified");
  }

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};
  if (optionsac.argc > 0) {
    if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
      return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified");
    }

    const char* matches[] = {"OUTPUTS"};
    AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);

    if (!AC_IsAtEnd(&optionsac)) {
      if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
        return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified");
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

  RAI_ModelOpts opts = {
    .batchsize = batchsize,
    .minbatchsize = minbatchsize
  };

  RAI_Model *model = NULL;

  size_t modellen;
  const char *modeldef;
  AC_GetString(&ac, &modeldef, &modellen, 0); 

  RAI_Error err = {0};

  model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend\n", bckstr);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "could not load %s default backend\n", bckstr);
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);
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

  if (ensureRunQueue(devicestr) == REDISMODULE_ERR) {
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
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

  model->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_MODEL, backend, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/**
* AI.MODELGET model_key [META | BLOB]
*/
int RedisAI_ModelGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2 && argc != 3) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace( ctx, argv[1], &key, &mto, REDISMODULE_READ | REDISMODULE_WRITE);
  if (status == REDISMODULE_ERR) {
    return REDISMODULE_ERR;
  }
  int blob = 0;
  if(argc==3){
    const char *optstr = RedisModule_StringPtrLen(argv[2], NULL);
    if (!strcasecmp(optstr, "META")) {
      blob = 0;
    }
    else if (!strcasecmp(optstr, "BLOB")) {
      blob = 1;
    }
  }

  RAI_Error err = {0};

  char *buffer = NULL;
  size_t len = 0;

  if (blob) {
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
  }

  int outentries = blob ? 8 : 6;

  RedisModule_ReplyWithArray(ctx, outentries);

  RedisModule_ReplyWithSimpleString(ctx, "BACKEND");
  const char* backendstr = RAI_BackendName(mto->backend);
  RedisModule_ReplyWithSimpleString(ctx, backendstr);

  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, mto->devicestr);

  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, mto->tag ? mto->tag : "");

  if (blob) {
    RedisModule_ReplyWithSimpleString(ctx, "BLOB");
    RedisModule_ReplyWithStringBuffer(ctx, buffer, len);
    RedisModule_Free(buffer);
  }
  RedisModule_CloseKey(key);
  return REDISMODULE_OK;
}

/**
* AI.MODELDEL model_key
*/
int RedisAI_ModelDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &key, &mto, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);
  RedisModule_ReplicateVerbatim(ctx);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI._MODELLIST
*/
int RedisAI_ModelList_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "MODELLIST is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_MODEL, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithSimpleString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/**
 * AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 *
 * 1. clone inputs as needed in the main thread (only the alternative is to
 * lock)
 * 2. spawn the new thread for running the model
 * 3. have reply callback put the data back into the key
 * 
 * This way we avoid any race condition. The only gotcha is making sure no one
 * overwrites the model until it's done computing.
 * This means that setModel will decode on a candidate pointer, and will then
 * be picked up on the next round. We also need to signal when it's time to
 * dispose of the old model. The key is having a single thread looping
 * forexecution
 */
int RedisAI_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                  int argc) {
  if (argc < 3) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *modelKey;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &modelKey, &mto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  const char *inputstr = RedisModule_StringPtrLen(argv[2], NULL);
  if (strcasecmp(inputstr, "INPUTS")) {
    RedisModule_CloseKey(modelKey);
    return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified");
  }

  struct RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  RedisModule_RetainString(NULL, argv[1]);
  rinfo->runkey = argv[1];
  rinfo->mctx = RAI_ModelRunCtxCreate(mto);
  // rinfo->mctxs = array_new(RAI_ModelRunCtx, 10);
  // rinfo->mctxs = array_append(rinfo->mctxs, );
  rinfo->sctx = NULL;
  rinfo->outkeys = NULL;
  rinfo->err = NULL;

  // parsing aux vars
  int is_input = 0;
  size_t ninputs = 0;
  size_t noutputs = 0;
  int outputs_flag_count = 0;

  for (size_t argpos = 3; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "OUTPUTS") && outputs_flag_count == 0) {
      is_input = 1;
      outputs_flag_count = 1;
      const size_t expected_noutputs = argc - argpos - 1;
      if(expected_noutputs>0){
        rinfo->outkeys =
          RedisModule_Calloc(expected_noutputs, sizeof(RedisModuleString *));
      }
    } else {
      RedisModule_RetainString(NULL, argv[argpos]);
      if (is_input == 0) {
        RAI_Tensor *inputTensor;
        RedisModuleKey *tensorKey;
        const int status = RAI_GetTensorFromKeyspace(
            ctx, argv[argpos], &tensorKey, &inputTensor, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          // TODO: free rinfo
          RedisModule_CloseKey(modelKey);
          return REDISMODULE_ERR;
        }
        RedisModule_CloseKey(tensorKey);
        // Opname here is passed without copying
        const char *opname = NULL;
        if (mto->inputs) {
          opname = mto->inputs[ninputs];
        }
        if (!RAI_ModelRunCtxAddInput(rinfo->mctx, opname, inputTensor)) {
          // todo free rinfo
          return RedisModule_ReplyWithError(ctx, "ERR Input key not found");
        }
        ninputs++;
      } else {
        // Opname here is passed without copying
        const char *opname = NULL;
        if (mto->outputs) {
          opname = mto->outputs[noutputs];
        }
        if (!RAI_ModelRunCtxAddOutput(rinfo->mctx, opname)) {
          // todo free rinfo
          return RedisModule_ReplyWithError(ctx, "ERR Output key not found");
        }
        rinfo->outkeys[noutputs] = argv[argpos];
        noutputs++;
      }
    }
  }

  if (mto->inputs && array_len(mto->inputs) != ninputs) {
    return RedisModule_ReplyWithError(
        ctx,
        "Number of names given as INPUTS during MODELSET and keys given as INPUTS here do not match");
  }

  if (mto->outputs && array_len(mto->outputs) != noutputs) {
    return RedisModule_ReplyWithError(
        ctx,
        "Number of names given as OUTPUTS during MODELSET and keys given as OUTPUTS here do not match");
  }

  AI_dictEntry *entry = AI_dictFind(run_queues, mto->devicestr);
  RunQueueInfo *run_queue_info = NULL;
  if (!entry) {
    // If the queue does not exist, initialize it
    if (ensureRunQueue(mto->devicestr) == REDISMODULE_ERR) {
      return RedisModule_ReplyWithError(ctx, "ERR Queue not initialized for device");
    }
    entry = AI_dictFind(run_queues, mto->devicestr);
    run_queue_info = AI_dictGetVal(entry);
  } else {
    run_queue_info = AI_dictGetVal(entry);
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_Run_Reply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

/** 
* AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
*/
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

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  const char* fnname;
  AC_GetString(&ac, &fnname, NULL, 0); 

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};

  if (!AC_AdvanceIfMatch(&ac, "INPUTS")) {
    return RedisModule_ReplyWithError(ctx, "INPUTS not specified");
  }

  const char* matches[] = {"OUTPUTS"};
  AC_GetSliceUntilMatches(&ac, &inac, 1, matches);

  if (!AC_AdvanceIfMatch(&ac, "OUTPUTS")) {
    return RedisModule_ReplyWithError(ctx, "OUTPUTS not specified");
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

  RAI_ScriptRunCtx *sctx = RAI_ScriptRunCtxCreate(sto, fnname);

  RedisModuleString **outkeys;

  for (size_t i=0; i<ninputs; i++) {
    RAI_Tensor *t;
    RedisModuleKey *argkey;
    const int status = RAI_GetTensorFromKeyspace(ctx, inputs[i], &argkey, &t, REDISMODULE_READ);
    if(status==REDISMODULE_ERR){
         RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }
    RedisModule_CloseKey(argkey);
    if (!RAI_ScriptRunCtxAddInput(sctx, t)) {
      RAI_ScriptRunCtxFree(sctx);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "Input key not found");
    }
  }

  outkeys = RedisModule_Calloc(noutputs, sizeof(RedisModuleString*));
  for (size_t i=0; i<noutputs; i++) {
    if (!RAI_ScriptRunCtxAddOutput(sctx)) {
      RAI_ScriptRunCtxFree(sctx);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "Output key not found");
    }
    RedisModule_RetainString(ctx, outputs[i]);
    outkeys[i] = outputs[i];
  }

  struct RedisAI_RunInfo *rinfo = RedisModule_Calloc(1, sizeof(struct RedisAI_RunInfo));
  rinfo->mctx = NULL;
  rinfo->sctx = sctx;
  RedisModule_RetainString(ctx, keystr);
  rinfo->runkey = keystr;
  rinfo->outkeys = outkeys;
  rinfo->err = NULL;
  AI_dictEntry *entry = AI_dictFind(run_queues, sto->devicestr);
  RunQueueInfo *run_queue_info = NULL;
  if (!entry){
    RAI_ScriptRunCtxFree(sctx);
    return RedisModule_ReplyWithError(ctx, "Queue not initialized for device");
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
  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTGET script_key
 */
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_ReplyWithArray(ctx, 6);
  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, sto->devicestr);
  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, sto->tag);
  RedisModule_ReplyWithSimpleString(ctx, "SOURCE");
  RedisModule_ReplyWithSimpleString(ctx, sto->scriptdef);
  RedisModule_CloseKey(key);
  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTDEL script_key
 */
int RedisAI_ScriptDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);

  RedisModule_ReplicateVerbatim(ctx);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI.SCRIPTSET script_key device [TAG tag] script_source
*/
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 4 && argc != 6) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "Insufficient arguments, missing script definition");
  }

  RAI_Script *script = NULL;

  size_t scriptlen;
  const char *scriptdef;
  AC_GetString(&ac, &scriptdef, &scriptlen, 0); 

  RAI_Error err = {0};
  script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "Backend TORCH not loaded, will try loading default backend\n");
    int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load TORCH default backend\n");
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
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
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
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

  script->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/** 
* AI._SCRIPTLIST
*/
int RedisAI_ScriptList_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "SCRIPTLIST is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_SCRIPT, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithSimpleString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/** 
* AI.INFO <model_or_script_key> [RESETSTAT]
*/
int RedisAI_Info_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 2 && argc != 3) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  const char* runkey;
  AC_GetString(&ac, &runkey, NULL, 0); 

  AI_dictEntry *stats_entry = AI_dictFind(run_stats, runkey);

  if (!stats_entry) {
    return RedisModule_ReplyWithError(ctx, "ERR cannot find run info for key");
  }

  struct RedisAI_RunStats *rstats = AI_dictGetVal(stats_entry);

  if (!AC_IsAtEnd(&ac)) {
    const char* opt;
    AC_GetString(&ac, &opt, NULL, 0); 

    if (strcasecmp(opt, "RESETSTAT") == 0) {
      rstats->duration_us = 0;
      rstats->samples = 0;
      rstats->calls = 0;
      rstats->nerrors = 0;
      RedisModule_ReplyWithSimpleString(ctx, "OK");
      return REDISMODULE_OK;
    }
  }

  RedisModule_ReplyWithArray(ctx, 18);

  RedisModule_ReplyWithSimpleString(ctx, "KEY");
  RedisModule_ReplyWithString(ctx, rstats->key);
  RedisModule_ReplyWithSimpleString(ctx, "TYPE");
  if (rstats->type == 0) {
    RedisModule_ReplyWithSimpleString(ctx, "MODEL");
  }
  else {
    RedisModule_ReplyWithSimpleString(ctx, "SCRIPT");
  }
  RedisModule_ReplyWithSimpleString(ctx, "BACKEND");
  RedisModule_ReplyWithSimpleString(ctx, RAI_BackendName(rstats->backend));
  RedisModule_ReplyWithSimpleString(ctx, "DEVICE");
  RedisModule_ReplyWithSimpleString(ctx, rstats->devicestr);
  RedisModule_ReplyWithSimpleString(ctx, "TAG");
  RedisModule_ReplyWithSimpleString(ctx, rstats->tag);
  RedisModule_ReplyWithSimpleString(ctx, "DURATION");
  RedisModule_ReplyWithLongLong(ctx, rstats->duration_us);
  RedisModule_ReplyWithSimpleString(ctx, "SAMPLES");
  if (rstats->type == 0) {
    RedisModule_ReplyWithLongLong(ctx, rstats->samples);
  }
  else {
    RedisModule_ReplyWithLongLong(ctx, -1);
  }
  RedisModule_ReplyWithSimpleString(ctx, "CALLS");
  RedisModule_ReplyWithLongLong(ctx, rstats->calls);
  RedisModule_ReplyWithSimpleString(ctx, "ERRORS");
  RedisModule_ReplyWithLongLong(ctx, rstats->nerrors);

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
  else if (strcasecmp(backend, "TFLITE") == 0) {
    ret = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, path);
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

/** 
* AI.CONFIG [BACKENDSPATH <default_location_of_backend_libraries> | LOADBACKEND <backend_identifier> <location_of_backend_library>]
*/
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
    RedisModule_Log(ctx, "warning", "Redis version does not support SharedAPI; running without exposing C API to other modules");
  }

  REGISTER_API(GetLLAPIVersion, ctx);

  REGISTER_API(TensorCreate, ctx);
  REGISTER_API(TensorDataSize, ctx);
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

  if (RedisModule_CreateCommand(ctx, "ai._modellist", RedisAI_ModelList_RedisCommand, "readonly", 1, 1, 1)
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

  if (RedisModule_CreateCommand(ctx, "ai._scriptlist", RedisAI_ScriptList_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.info", RedisAI_Info_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.config", RedisAI_Config_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (argc > 0 && argc % 2 != 0) {
    RedisModule_Log(ctx, "warning", "Even number of arguments provided to module. Please provide arguments as KEY VAL pairs");
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
    else if (strcasecmp(key, "TFLITE") == 0) {
      ret = RAI_LoadBackend(ctx, RAI_BACKEND_TFLITE, val);
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

  run_stats = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  
  return REDISMODULE_OK;
}

extern AI_dictType AI_dictTypeHeapStrings;
