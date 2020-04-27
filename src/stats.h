/**
 * stats.h
 *
 * Contains the structure and headers for the helper methods to create,
 * initialize, get, reset, and free run-time statics, like call count, error
 * count, and aggregate durations of ModelRun and ScriptRun sessions.
 *
 */

#ifndef SRC_STATS_H_
#define SRC_STATS_H_

#include <sys/time.h>

#include "config.h"
#include "redismodule.h"
#include "util/dict.h"

struct RedisAI_RunStats {
  RedisModuleString* key;
  RAI_RunType type;
  RAI_Backend backend;
  char* devicestr;
  char* tag;
  long long duration_us;
  long long samples;
  long long calls;
  long long nerrors;
};

AI_dict* run_stats;

long long ustime(void);
mstime_t mstime(void);

/**
 * Adds an entry to the ephemeral run-time statistic. The statistics are not
 * saved to the keyspace, and on maximum live for the duration of the DB uptime.
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name to use as unique stats identifier
 * @param type type of stats identifier ( one of RAI_MODEL or RAI_SCRIPT )
 * @param backend backend identifier (one of RAI_BACKEND_TENSORFLOW,
 * RAI_BACKEND_TFLITE, RAI_BACKEND_TORCH, RAI_BACKEND_ONNXRUNTIME,)
 * @param devicestr
 * @param tag optional tag of Model/Script
 * @return
 */
void* RAI_AddStatsEntry(RedisModuleCtx* ctx, RedisModuleString* key,
                        RAI_RunType type, RAI_Backend backend,
                        const char* devicestr, const char* tag);

/**
 * Removes the statistical entry with the provided unique stats identifier
 *
 * @param infokey
 */
void RAI_RemoveStatsEntry(void* infokey);

/**
 * Returns a list of all statistical entries that match a specific RAI_RunType (
 * model or script )
 *
 * @param type type of stats identifier to provide the list for ( one of
 * RAI_MODEL or RAI_SCRIPT )
 * @param nkeys output variable containing the number of returned stats
 * @param keys output variable containing the list of returned keys
 * @param tags output variable containing the list of returned tags
 */
void RAI_ListStatsEntries(RAI_RunType type, long long* nkeys,
                          RedisModuleString*** keys, const char*** tags);

/**
 * Frees the memory of the RedisAI_RunStats, excluding the one managed by the
 * Context in which Redis modules operate
 *
 * @param rstats
 */
void RAI_FreeRunStats(struct RedisAI_RunStats* rstats);

/**
 * Frees the memory of the RedisAI_RunStats, including the one managed by the
 * Context in which Redis modules operate
 *
 * @param ctx Context in which Redis modules operate
 * @param rstats
 */
void RedisAI_FreeRunStats(RedisModuleCtx* ctx, struct RedisAI_RunStats* rstats);

#endif /* SRC_SATTS_H_ */
