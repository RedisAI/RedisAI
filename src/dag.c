/**
 * dag.c
 *
 * Contains the helper methods for both parsing, running the command in the
 * background, and replying DAG structured commands.
 */

#include "dag.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

/**
 * Actual method running the DAGRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RedisAI_DagRunSession(RedisAI_RunInfo *rinfo) {
  for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET: {
        RAI_Tensor *t = NULL;
        const int parse_result = RAI_parseTensorSetArgs(
            NULL, currentOp->argv, currentOp->argc, &t, 0, currentOp->err);
        if (parse_result > 0) {
          const char *key_string =
              RedisModule_StringPtrLen(currentOp->argv[1], NULL);
          const char *dictKey = RedisModule_Strdup(key_string);
          AI_dictReplace(rinfo->dagTensorsContext, (void*)dictKey, t);
          currentOp->result = REDISMODULE_OK;
        } else {
          currentOp->result = REDISMODULE_ERR;
        }
        break;
      }
      case REDISAI_DAG_CMD_TENSORGET: {
        const char *key_string =
            RedisModule_StringPtrLen(currentOp->argv[1], NULL);
        RAI_Tensor *t = NULL;
        currentOp->result = RAI_getTensorFromLocalContext(
            NULL, rinfo->dagTensorsContext, key_string, &t, currentOp->err);
        if (currentOp->result == REDISMODULE_OK) {
          RAI_Tensor *outTensor = NULL;
          // TODO: check tensor copy return value
            RAI_TensorDeepCopy(t, &outTensor);
          array_append(currentOp->outTensors, outTensor);
          currentOp->result = REDISMODULE_OK;
        }
        break;
      }
      case REDISAI_DAG_CMD_MODELRUN: {
        const int parse_result = RedisAI_Parse_ModelRun_RedisCommand(
            NULL, currentOp->argv, currentOp->argc, &(currentOp->mctx),
            &(currentOp->outkeys), &(currentOp->mctx->model), 1,
            &(rinfo->dagTensorsContext), 0, NULL, currentOp->err);

        if (parse_result > 0) {
          RAI_ModelRunCtx *mctxs[1];
          mctxs[0] = currentOp->mctx;
          currentOp->result = REDISMODULE_OK;
          const long long start = ustime();
          currentOp->result = RAI_ModelRun(mctxs, 1, currentOp->err);
          currentOp->duration_us = ustime() - start;
          const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
          for (size_t outputNumber = 0; outputNumber < noutputs;
               outputNumber++) {
            RAI_Tensor *tensor =
                RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
            if (tensor) {
              const char *key_string = RedisModule_StringPtrLen(
                  currentOp->outkeys[outputNumber], NULL);
              const char *dictKey = RedisModule_Strdup(key_string);
              AI_dictReplace(rinfo->dagTensorsContext, (void*)dictKey, RAI_TensorGetShallowCopy(tensor));
            } else {
              RAI_SetError(currentOp->err, RAI_EMODELRUN,
                           "ERR output tensor on DAG's MODELRUN was null");
              currentOp->result = REDISMODULE_ERR;
            }
          }
          // since we've increased the reference count prior modelrun we need to decrease it
          const size_t ninputs = RAI_ModelRunCtxNumInputs(currentOp->mctx);
          for (size_t inputNumber = 0; inputNumber < ninputs; inputNumber++) {
            RAI_Tensor *tensor =
                RAI_ModelRunCtxInputTensor(currentOp->mctx, inputNumber);
            if (tensor) {
              RAI_TensorFree(tensor);
            }
          }

        } else {
          currentOp->result = REDISMODULE_ERR;
        }
        break;
      }
        case REDISAI_DAG_CMD_SCRIPTRUN: {
            const int parse_result = RedisAI_Parse_ScriptRun_RedisCommand(
                    NULL, currentOp->argv, currentOp->argc, &(currentOp->sctx),
                    &(currentOp->outkeys), &(currentOp->sctx->script), 1,
                    &(rinfo->dagTensorsContext), 0, NULL, currentOp->err);

            if (parse_result > 0) {
                currentOp->result = REDISMODULE_OK;
                const long long start = ustime();
                currentOp->result = RAI_ScriptRun(currentOp->sctx, currentOp->err);
                currentOp->duration_us = ustime() - start;
                const size_t noutputs = RAI_ScriptRunCtxNumOutputs(currentOp->sctx);
                for (size_t outputNumber = 0; outputNumber < noutputs;
                     outputNumber++) {
                    RAI_Tensor *tensor =
                            RAI_ScriptRunCtxOutputTensor(currentOp->sctx, outputNumber);
                    if (tensor) {
                        const char *key_string = RedisModule_StringPtrLen(
                                currentOp->outkeys[outputNumber], NULL);
                        const char *dictKey = RedisModule_Strdup(key_string);
                        AI_dictReplace(rinfo->dagTensorsContext, (void*)dictKey, RAI_TensorGetShallowCopy(tensor));
                    } else {
                        RAI_SetError(currentOp->err, RAI_EMODELRUN,
                                     "ERR output tensor on DAG's SCRIPTRUN was null");
                        currentOp->result = REDISMODULE_ERR;
                    }
                }
            } else {
                currentOp->result = REDISMODULE_ERR;
            }
            break;
        }
      default: {
        /* unsupported DAG's command */
        RAI_SetError(currentOp->err, RAI_EDAGRUN,
                     "ERR unsupported command within DAG");
        currentOp->result = REDISMODULE_ERR;
        break;
      }
    }
  }
  if (rinfo->client != NULL) {
    RedisModule_UnblockClient(rinfo->client, rinfo);
  }
  return NULL;
}

int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);
  RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
  for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      case REDISAI_DAG_CMD_TENSORGET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          if (array_len(currentOp->outTensors) > 0) {
            RAI_Tensor *tensor = currentOp->outTensors[0];
            RAI_parseTensorGetArgs(ctx, currentOp->argv, currentOp->argc,
                                   tensor);
          } else {
            RedisModule_ReplyWithError(
                ctx, "ERR error getting tensor from local context");
          }
        }
        break;
      }

      case REDISAI_DAG_CMD_MODELRUN: {
        rinfo->dagReplyLength++;
        struct RedisAI_RunStats *rstats = NULL;
        const char *runkey =
            RedisModule_StringPtrLen(currentOp->runkey, NULL);
        RAI_GetRunStats(runkey,&rstats);
        if (currentOp->result == REDISMODULE_ERR) {
          RAI_SafeAddDataPoint(rstats,0,1,1,0);
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          RAI_SafeAddDataPoint(rstats,currentOp->duration_us,1,0,0);
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

        case REDISAI_DAG_CMD_SCRIPTRUN: {
            rinfo->dagReplyLength++;
            struct RedisAI_RunStats *rstats = NULL;
            const char *runkey =
                    RedisModule_StringPtrLen(currentOp->runkey, NULL);
            RAI_GetRunStats(runkey,&rstats);
            if (currentOp->result == REDISMODULE_ERR) {
                RAI_SafeAddDataPoint(rstats,0,1,1,0);
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
            } else {
                RAI_SafeAddDataPoint(rstats,currentOp->duration_us,1,0,0);
                RedisModule_ReplyWithSimpleString(ctx, "OK");
            }
            break;
        }

      default:
        /* no-op */
        break;
    }
  }

  AI_dictIterator *persist_iter =
      AI_dictGetSafeIterator(rinfo->dagTensorsPersistentContext);
  AI_dictEntry *persist_entry = AI_dictNext(persist_iter);
  while (persist_entry) {
    const char *persist_key_name = AI_dictGetKey(persist_entry);
    AI_dictEntry *tensor_entry =
        AI_dictFind(rinfo->dagTensorsContext, persist_key_name);
    if (tensor_entry) {
      RAI_Tensor *tensor = RAI_TensorGetShallowCopy(AI_dictGetVal(tensor_entry));
      RedisModuleKey *key;
      RedisModuleString *tensor_keyname = RedisModule_CreateString(
          ctx, persist_key_name, strlen(persist_key_name));
      const int status = RAI_OpenKey_Tensor(
          ctx, tensor_keyname, &key, REDISMODULE_READ | REDISMODULE_WRITE);
      if (status == REDISMODULE_ERR) {
        RAI_TensorFree(tensor);
        RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
        rinfo->dagReplyLength++;
      } else {
        if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, tensor) !=
            REDISMODULE_OK) {
          RAI_TensorFree(tensor);
          RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
          rinfo->dagReplyLength++;
        }
      }
      RedisModule_CloseKey(key);
      RedisAI_ReplicateTensorSet(ctx, tensor_keyname, tensor);
    } else {
      RedisModule_ReplyWithError(
          ctx, "ERR specified persistent key that was not used on DAG");
      rinfo->dagReplyLength++;

      RedisModule_Log(ctx, "warning",
                      "on DAGRUN's PERSIST pecified persistent key (%s) that "
                      "was not used on DAG. Logging all local context keys",
                      persist_key_name);
      AI_dictIterator *local_iter =
          AI_dictGetSafeIterator(rinfo->dagTensorsContext);
      AI_dictEntry *local_entry = AI_dictNext(local_iter);
      while (local_entry) {
        const char *localcontext_key_name = AI_dictGetKey(local_entry);
        RedisModule_Log(ctx, "warning", "DAG's local context key (%s)",
                        localcontext_key_name);
        local_entry = AI_dictNext(local_iter);
      }
      AI_dictReleaseIterator(local_iter);

      for (size_t opN = 0; opN < array_len(rinfo->dagOps); opN++) {
        RedisModule_Log(
            ctx, "warning", "DAG's op n#  %zu - cmdType %d ( argc %d )", opN,
            rinfo->dagOps[opN]->commandType, rinfo->dagOps[opN]->argc);
      }
    }

    persist_entry = AI_dictNext(persist_iter);
  }
  AI_dictReleaseIterator(persist_iter);
  RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
  RAI_FreeRunInfo(ctx, rinfo);
  return REDISMODULE_OK;
}

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc, AI_dict **loadedContextDict,
                         AI_dict **localContextDict,
                         const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx, "ERR invalid or negative value found in number of keys to LOAD");
    return -1;
  }
  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos <= argc - 1) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      RAI_Tensor *t;
      RedisModuleKey *key;
      const int status = RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t, REDISMODULE_READ);
      if (status == REDISMODULE_ERR) {
        RedisModule_Log(
            ctx, "warning",
            "on DAGRUN's LOAD could not load tensor %s from keyspace",
            arg_string);
        return -1;
      }
      RedisModule_CloseKey(key);
      const char *dictKey = RedisModule_Strdup(arg_string);
      AI_dictAdd(*localContextDict, (void*)dictKey, RAI_TensorGetShallowCopy(t));
      const char *keyspacePersistKey = RedisModule_Strdup(dictKey);
      AI_dictAdd(*loadedContextDict, (void*)keyspacePersistKey, (void *)1);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                            int argc, AI_dict **persistContextDict,
                            const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx,
        "ERR invalid or negative value found in number of keys to PERSIST");
    return -1;
  }

  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos < argc) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      const char *key = RedisModule_Strdup(arg_string);
      AI_dictAdd(*persistContextDict, (void*)key, (void *)1);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}

int RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                    RedisModuleString **argv, int argc){
    for (size_t argpos = 1; argpos < argc; argpos++){
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if ( (!strcasecmp(arg_string, "LOAD") || !strcasecmp(arg_string, "PERSIST") ) && (argpos+1 < argc) ) {
            long long n_keys;
            argpos++;
            const int retval = RedisModule_StringToLongLong(argv[argpos], &n_keys);
            if(retval != REDISMODULE_OK){
                return REDISMODULE_ERR;
            }
            argpos++;
            if (n_keys > 0){
                size_t last_persist_argpos = n_keys+argpos;
                for (; argpos < last_persist_argpos &&  argpos < argc; argpos++){
                    RedisModule_KeyAtPos(ctx, argpos);
                }
            }
        }
    }
    return REDISMODULE_OK;
}

int RedisAI_DagRunSyntaxParser(RedisModuleCtx *ctx, RedisModuleString **argv,
                                 int argc, int dagMode) {
  if (RedisModule_IsKeysPositionRequest(ctx)) {
     return RedisAI_DagRun_IsKeysPositionRequest_ReportKeys(ctx, argv, argc);
  }
  if (argc < 4) return RedisModule_WrongArity(ctx);
  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(
        ctx,
        "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo "
        "structure");
  }
  rinfo->use_local_context = 1;
  RAI_DagOp *currentDagOp = NULL;
  RAI_InitDagOp(&currentDagOp);
  rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);

  int persistFlag = 0;
  int loadFlag = 0;
  int chainingOpCount = 0;
  const char *deviceStr = NULL;

  for (size_t argpos = 1; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "LOAD")) {
      loadFlag = 1;
      const int parse_result = RAI_parseDAGLoadArgs(
          ctx, &argv[argpos], argc - argpos, &(rinfo->dagTensorsLoadedContext),
          &(rinfo->dagTensorsContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx, rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "PERSIST")) {
      if (dagMode == REDISAI_DAG_READONLY_MODE) {
        RAI_FreeRunInfo(ctx, rinfo);
        return RedisModule_ReplyWithError(
            ctx, "ERR PERSIST cannot be specified in a read-only DAG");
      }
      persistFlag = 1;
      const int parse_result =
          RAI_parseDAGPersistArgs(ctx, &argv[argpos], argc - argpos,
                                  &(rinfo->dagTensorsPersistentContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx, rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "|>")) {
      // on the first pipe operator, if LOAD or PERSIST were used, we've already
      // allocated memory
      if (!((persistFlag == 1 || loadFlag == 1) && chainingOpCount == 0)) {
        rinfo->dagNumberCommands++;
        RAI_DagOp *currentDagOp = NULL;
        RAI_InitDagOp(&currentDagOp);
        rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
      }
      chainingOpCount++;
    } else {
      if (!strcasecmp(arg_string, "AI.TENSORGET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_TENSORGET;
      }
      if (!strcasecmp(arg_string, "AI.TENSORSET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_TENSORSET;
      }
      if (!strcasecmp(arg_string, "AI.MODELRUN")) {
        if (argc - 2 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_MODELRUN;
        RAI_Model *mto;
        RedisModuleKey *modelKey;
        const int status = RAI_GetModelFromKeyspace(
            ctx, argv[argpos + 1], &modelKey, &mto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx, rinfo);
          return REDISMODULE_ERR;
        }
        if (deviceStr == NULL) {
          deviceStr = mto->devicestr;
        } else {
          // If the device strings are not equivalent, reply with error ( for
          // now )
          if (strcasecmp(mto->devicestr, deviceStr) != 0) {
            RAI_FreeRunInfo(ctx, rinfo);
            return RedisModule_ReplyWithError(
                ctx, "ERR multi-device DAGs not supported yet");
          }
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->runkey = argv[argpos + 1];
        rinfo->dagOps[rinfo->dagNumberCommands]->mctx =
            RAI_ModelRunCtxCreate(mto);
      }
      if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
        if (argc - 3 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType =
            REDISAI_DAG_CMD_SCRIPTRUN;
        RAI_Script *sto;
        RedisModuleKey *scriptKey;
        const int status = RAI_GetScriptFromKeyspace(
            ctx, argv[argpos + 1], &scriptKey, &sto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx, rinfo);
          return REDISMODULE_ERR;
        }
        if (deviceStr == NULL) {
          deviceStr = sto->devicestr;
        } else {
          // If the device strings are not equivalent, reply with error ( for
          // now )
          if (strcasecmp(sto->devicestr, deviceStr) != 0) {
            RAI_FreeRunInfo(ctx, rinfo);
            return RedisModule_ReplyWithError(
                ctx, "ERR multi-device DAGs not supported yet");
          }
        }
        const char *functionName =
            RedisModule_StringPtrLen(argv[argpos + 2], NULL);
        rinfo->dagOps[rinfo->dagNumberCommands]->runkey = argv[argpos + 1];
        rinfo->dagOps[rinfo->dagNumberCommands]->sctx =
            RAI_ScriptRunCtxCreate(sto, functionName);
      }
      RedisModule_RetainString(NULL, argv[argpos]);
      array_append(rinfo->dagOps[rinfo->dagNumberCommands]->argv, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argc++;
    }
  }

  RunQueueInfo *run_queue_info = NULL;
  // If there was no MODELRUN or SCRIPTRUN on the DAG, we default all ops to CPU
  if (deviceStr == NULL) {
    deviceStr = "CPU";
  }
  // If the queue does not exist, initialize it
  if (ensureRunQueue(deviceStr, &run_queue_info) == REDISMODULE_ERR) {
    RAI_FreeRunInfo(ctx, rinfo);
    return RedisModule_ReplyWithError(ctx,
                                      "ERR Queue not initialized for device");
  }

  rinfo->client =
      RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, NULL, 0);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}
