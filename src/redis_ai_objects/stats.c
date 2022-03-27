/**
 * stats.c
 *
 * Contains the helper methods to create,
 * initialize, get, reset, and free run-time statistics, like call count, error
 * count, and aggregate durations of ModelRun and ScriptRun sessions.
 *
 */

#include <sys/time.h>
#include <stdlib.h>
#include "stats.h"
#include "util/string_utils.h"

// Global dictionary that stores run statistics for all models and scripts in the shard.
AI_dict *RunStats;

long long ustime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust;
}

mstime_t mstime(void) { return ustime() / 1000; }

RAI_RunStats *RAI_StatsCreate(RedisModuleString *key, RAI_RunType type, RAI_Backend backend,
                              const char *device_str, RedisModuleString *tag) {
    RAI_RunStats *r_stats = RedisModule_Calloc(1, sizeof(RAI_RunStats));
    r_stats->key = RedisModule_CreateStringFromString(NULL, key);
    r_stats->type = type;
    r_stats->backend = backend;
    r_stats->device_str = RedisModule_Strdup(device_str);
    r_stats->tag = RAI_HoldString(tag);
    return r_stats;
}

void RAI_StatsReset(RAI_RunStats *r_stats) {
    RedisModule_Assert(r_stats);
    __atomic_store_n(&r_stats->duration_us, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&r_stats->samples, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&r_stats->calls, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&r_stats->n_errors, 0, __ATOMIC_RELAXED);
}

void RAI_StatsAddDataPoint(RAI_RunStats *r_stats, unsigned long duration, unsigned long calls,
                           unsigned long errors, unsigned long samples) {
    RedisModule_Assert(r_stats);
    __atomic_add_fetch(&r_stats->duration_us, duration, __ATOMIC_RELAXED);
    __atomic_add_fetch(&r_stats->calls, calls, __ATOMIC_RELAXED);
    __atomic_add_fetch(&r_stats->n_errors, errors, __ATOMIC_RELAXED);
    __atomic_add_fetch(&r_stats->samples, samples, __ATOMIC_RELAXED);
}

void RAI_StatsFree(RAI_RunStats *r_stats) {
    if (r_stats) {
        if (r_stats->device_str) {
            RedisModule_Free(r_stats->device_str);
        }
        if (r_stats->tag) {
            RedisModule_FreeString(NULL, r_stats->tag);
        }
        if (r_stats->key) {
            RedisModule_FreeString(NULL, r_stats->key);
        }
        RedisModule_Free(r_stats);
    }
}

/************************************* Global RunStats dict API *********************************/

void RAI_StatsStoreEntry(RedisModuleString *key, RAI_RunStats *new_stats_entry) {
    AI_dictReplace(RunStats, (void *)key, (void *)new_stats_entry);
}

void RAI_StatsGetAllEntries(RAI_RunType type, long long *nkeys, RedisModuleString ***keys,
                            RedisModuleString ***tags) {
    AI_dictIterator *stats_iter = AI_dictGetSafeIterator(RunStats);
    long long stats_size = AI_dictSize(RunStats);

    *keys = RedisModule_Calloc(stats_size, sizeof(RedisModuleString *));
    *tags = RedisModule_Calloc(stats_size, sizeof(RedisModuleString *));
    *nkeys = 0;

    AI_dictEntry *stats_entry = AI_dictNext(stats_iter);
    RAI_RunStats *r_stats = NULL;

    while (stats_entry) {
        r_stats = AI_dictGetVal(stats_entry);
        if (r_stats->type == type) {
            (*keys)[*nkeys] = r_stats->key;
            (*tags)[*nkeys] = r_stats->tag;
            *nkeys += 1;
        }
        stats_entry = AI_dictNext(stats_iter);
    }
    AI_dictReleaseIterator(stats_iter);
}

void RAI_StatsRemoveEntry(RedisModuleString *info_key) {
    AI_dictEntry *stats_entry = AI_dictFind(RunStats, info_key);

    if (stats_entry) {
        RAI_RunStats *r_stats = AI_dictGetVal(stats_entry);
        AI_dictDelete(RunStats, info_key);
        RAI_StatsFree(r_stats);
    }
}

RAI_RunStats *RAI_StatsGetEntry(RedisModuleString *runkey) {
    RedisModule_Assert(RunStats);
    AI_dictEntry *entry = AI_dictFind(RunStats, runkey);
    if (!entry) {
        return NULL;
    }
    return AI_dictGetVal(entry);
}
