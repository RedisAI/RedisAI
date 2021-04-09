/**
 * stats.h
 *
 * Contains the structure and headers for the helper methods to create,
 * initialize, get, reset, and free run-time statics, like call count, error
 * count, and aggregate durations of ModelRun and ScriptRun sessions.
 *
 */

#pragma once

#include "config/config.h"
#include "redismodule.h"
#include "util/dict.h"

typedef struct RedisAI_RunStats {
    RedisModuleString *key;
    RAI_RunType type;
    RAI_Backend backend;
    char *devicestr;
    RedisModuleString *tag;
    long long duration_us;
    long long samples;
    long long calls;
    long long nerrors;
} RedisAI_RunStats;

AI_dict *run_stats;

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
void *RAI_AddStatsEntry(RedisModuleCtx *ctx, RedisModuleString *key, RAI_RunType type,
                        RAI_Backend backend, const char *devicestr, RedisModuleString *tag);

/**
 * Removes the statistical entry with the provided unique stats identifier
 *
 * @param infokey
 */
void RAI_RemoveStatsEntry(void *infokey);

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
void RAI_ListStatsEntries(RAI_RunType type, long long *nkeys, RedisModuleString ***keys,
                          RedisModuleString ***tags);

/**
 *
 * @param rstats
 * @return 0 on success, or 1 if the reset failed
 */
int RAI_ResetRunStats(struct RedisAI_RunStats *rstats);

/**
 * Safely add datapoint to the run stats. Protected against null pointer
 * runstats
 * @param rstats
 * @param duration
 * @param calls
 * @param errors
 * @param samples
 * @return 0 on success, or 1 if the addition failed
 */
int RAI_SafeAddDataPoint(struct RedisAI_RunStats *rstats, long long duration, long long calls,
                         long long errors, long long samples);

void RAI_FreeRunStats(struct RedisAI_RunStats *rstats);

/**
 *
 * @param runkey
 * @param rstats
 * @return 0 on success, or 1 if the the run stats with runkey does not exist
 */
int RAI_GetRunStats(RedisModuleString *runkey, struct RedisAI_RunStats **rstats);

void RedisAI_FreeRunStats(RedisModuleCtx *ctx, struct RedisAI_RunStats *rstats);
