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

typedef struct RAI_RunStats {
    RedisModuleString *key;
    RAI_RunType type;
    RAI_Backend backend;
    char *device_str;
    RedisModuleString *tag;
    unsigned long duration_us;
    unsigned long samples;
    unsigned long calls;
    unsigned long n_errors;
} RAI_RunStats;

long long ustime(void);
mstime_t mstime(void);

/**
 * Adds an entry to the ephemeral run-time statistic. The statistics are not
 * saved to the keyspace, and on maximum live for the duration of the DB uptime.
 *
 * @param key key name to use as unique stats identifier
 * @param type type of stats identifier ( one of RAI_MODEL or RAI_SCRIPT )
 * @param backend backend identifier (one of RAI_BACKEND_TENSORFLOW,
 * RAI_BACKEND_TFLITE, RAI_BACKEND_TORCH, RAI_BACKEND_ONNXRUNTIME,)
 * @param device_str device to execute the model on (CPU, GPU, ...)
 * @param tag optional tag of Model/Script
 * @return A newly heap allocated RedisAI_RunStats object with the given fields.
 */
RAI_RunStats *RAI_StatsCreate(RedisModuleString *key, RAI_RunType type, RAI_Backend backend,
                              const char *device_str, RedisModuleString *tag);

/**
 * @brief Reset atomically counters for a given run_stats of some model/script.
 * @param run_stats entry to reset.
 */
void RAI_StatsReset(RAI_RunStats *run_stats);

/**
 * Update atomically stats counters after execution.
 * @param r_stats runStats entry that matches some model/script.
 * @param duration execution runtime in us
 * @param calls number of calls to the underline model/script operation.
 * @param errors number of errors that had occurred.
 * @param samples number of samples that the model execute (batch size)
 */
void RAI_StatsAddDataPoint(RAI_RunStats *r_stats, unsigned long duration, unsigned long calls,
                           unsigned long errors, unsigned long samples);

/**
 * @brief Release RunStats struct.
 * @param run_stats entry to remove.
 */
void RAI_StatsFree(RAI_RunStats *r_stats);

/************************************* Global RunStats dict API *********************************/
/**
 * Adds an entry to the ephemeral run-time statistic. The statistics are not
 * saved to the keyspace, and on maximum live for the duration of the DB uptime.
 *
 * @param keyName key name to use as unique stats identifier.
 * @param run_stats_entry RunStats entry pointer to store.
 */
void RAI_StatsStoreEntry(RedisModuleString *key, RAI_RunStats *run_stats_entry);

/**
 * @brief: Removes the statistical entry with the provided unique stats identifier
 * @param info_key
 */
void RAI_StatsRemoveEntry(RedisModuleString *info_key);

/**
 * Returns a list of all statistical entries that match a specific RAI_RunType (
 * model or script).
 * @param type type of stats identifier to provide the list for (one of
 * RAI_MODEL or RAI_SCRIPT).
 * @param nkeys output variable containing the number of returned stats.
 * @param keys output variable containing the list of returned keys.
 * @param tags output variable containing the list of returned tags.
 */
void RAI_StatsGetAllEntries(RAI_RunType type, long long *nkeys, RedisModuleString ***keys,
                            RedisModuleString ***tags);

/**
 * @brief Retrieve the run stats info of run_key from the global RunStat dictionary and set it in
 * r_stats.
 * @param run_key module/script key name
 * @param run_stats Place-holder for the RAI_RunStats object that is associated with the key.
 * @return REDISMODULE_OK on success, REDISMODULE_ERR if the the run stats with run_key does not
 * exist.
 */
int RAI_StatsGetEntry(RedisModuleString *run_key, RAI_RunStats **r_stats);
