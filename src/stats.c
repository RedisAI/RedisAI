/**
 * stats.c
 *
 * Contains the helper methods to create,
 * initialize, get, reset, and free run-time statics, like call count, error
 * count, and aggregate durations of ModelRun and ScriptRun sessions.
 *
 */

#include "stats.h"

#include <sys/time.h>

long long ustime(void) {
  struct timeval tv;
  long long ust;

  gettimeofday(&tv, NULL);
  ust = ((long long)tv.tv_sec) * 1000000;
  ust += tv.tv_usec;
  return ust;
}

mstime_t mstime(void) { return ustime() / 1000; }

void* RAI_AddStatsEntry(RedisModuleCtx* ctx, RedisModuleString* key,
                        RAI_RunType runtype, RAI_Backend backend,
                        const char* devicestr, const char* tag) {
  const char* infokey = RedisModule_StringPtrLen(key, NULL);

  struct RedisAI_RunStats* rstats = NULL;
  rstats = RedisModule_Calloc(1, sizeof(struct RedisAI_RunStats));
  RedisModule_RetainString(ctx, key);
  rstats->key = key;
  rstats->type = runtype;
  rstats->backend = backend;
  rstats->devicestr = RedisModule_Strdup(devicestr);
  rstats->tag = RedisModule_Strdup(tag);

  AI_dictAdd(run_stats, (void*)infokey, (void*)rstats);

  return (void*)infokey;
}

void RAI_ListStatsEntries(RAI_RunType type, long long* nkeys,
                          RedisModuleString*** keys, const char*** tags) {
  AI_dictIterator* stats_iter = AI_dictGetSafeIterator(run_stats);

  long long stats_size = AI_dictSize(run_stats);

  *keys = RedisModule_Calloc(stats_size, sizeof(RedisModuleString*));
  *tags = RedisModule_Calloc(stats_size, sizeof(const char*));

  *nkeys = 0;

  AI_dictEntry* stats_entry = AI_dictNext(stats_iter);
  struct RedisAI_RunStats* rstats = NULL;

  while (stats_entry) {
    rstats = AI_dictGetVal(stats_entry);

    if (rstats->type == type) {
      (*keys)[*nkeys] = rstats->key;
      (*tags)[*nkeys] = rstats->tag;
      *nkeys += 1;
    }

    stats_entry = AI_dictNext(stats_iter);
  }

  AI_dictReleaseIterator(stats_iter);
}

void RAI_RemoveStatsEntry(void* infokey) {
  AI_dictEntry* stats_entry = AI_dictFind(run_stats, infokey);

  if (stats_entry) {
    struct RedisAI_RunStats* rstats = AI_dictGetVal(stats_entry);
    AI_dictDelete(run_stats, infokey);
    RAI_FreeRunStats(rstats);
  }
}

void RAI_FreeRunStats(struct RedisAI_RunStats* rstats) {
  if (rstats) {
    if (rstats->devicestr) {
      RedisModule_Free(rstats->devicestr);
    }
    if (rstats->tag) {
      RedisModule_Free(rstats->tag);
    }
    RedisModule_Free(rstats);
  }
}

void RedisAI_FreeRunStats(RedisModuleCtx* ctx,
                          struct RedisAI_RunStats* rstats) {
  RedisModule_FreeString(ctx, rstats->key);
  RAI_FreeRunStats(rstats);
}