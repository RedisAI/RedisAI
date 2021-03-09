/**
 * stats.c
 *
 * Contains the helper methods to create,
 * initialize, get, reset, and free run-time statics, like call count, error
 * count, and aggregate durations of ModelRun and ScriptRun sessions.
 *
 */

#include <sys/time.h>
#include "stats.h"
#include "util/string_utils.h"


long long ustime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust;
}

mstime_t mstime(void) { return ustime() / 1000; }

void *RAI_AddStatsEntry(RedisModuleCtx *ctx, RedisModuleString *key, RAI_RunType runtype,
                        RAI_Backend backend, const char *devicestr, RedisModuleString *tag) {
    struct RedisAI_RunStats *rstats = NULL;
    rstats = RedisModule_Calloc(1, sizeof(struct RedisAI_RunStats));
    rstats->key = RAI_HoldString(NULL, key);
    rstats->type = runtype;
    rstats->backend = backend;
    rstats->devicestr = RedisModule_Strdup(devicestr);
    rstats->tag = RAI_HoldString(NULL, tag);

    AI_dictAdd(run_stats, (void *)key, (void *)rstats);

    return (void *)key;
}

void RAI_ListStatsEntries(RAI_RunType type, long long *nkeys, RedisModuleString ***keys,
                          RedisModuleString ***tags) {
    AI_dictIterator *stats_iter = AI_dictGetSafeIterator(run_stats);

    long long stats_size = AI_dictSize(run_stats);

    *keys = RedisModule_Calloc(stats_size, sizeof(RedisModuleString *));
    *tags = RedisModule_Calloc(stats_size, sizeof(RedisModuleString *));

    *nkeys = 0;

    AI_dictEntry *stats_entry = AI_dictNext(stats_iter);
    struct RedisAI_RunStats *rstats = NULL;

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

void RAI_RemoveStatsEntry(void *infokey) {
    AI_dictEntry *stats_entry = AI_dictFind(run_stats, infokey);

    if (stats_entry) {
        struct RedisAI_RunStats *rstats = AI_dictGetVal(stats_entry);
        AI_dictDelete(run_stats, infokey);
        RAI_FreeRunStats(rstats);
    }
}

int RAI_ResetRunStats(struct RedisAI_RunStats *rstats) {
    rstats->duration_us = 0;
    rstats->samples = 0;
    rstats->calls = 0;
    rstats->nerrors = 0;
    return 0;
}

int RAI_SafeAddDataPoint(struct RedisAI_RunStats *rstats, long long duration, long long calls,
                         long long errors, long long samples) {
    int result = 1;
    if (rstats == NULL) {
        return result;
    } else {
        rstats->duration_us += duration;
        rstats->calls += calls;
        rstats->nerrors += errors;
        rstats->samples += samples;
        result = 0;
    }
    return result;
}

void RAI_FreeRunStats(struct RedisAI_RunStats *rstats) {
    if (rstats) {
        if (rstats->devicestr) {
            RedisModule_Free(rstats->devicestr);
        }
        if (rstats->tag) {
            RedisModule_FreeString(NULL, rstats->tag);
        }
        if (rstats->key) {
            RedisModule_FreeString(NULL, rstats->key);
        }
        RedisModule_Free(rstats);
    }
}

int RAI_GetRunStats(RedisModuleString *runkey, struct RedisAI_RunStats **rstats) {
    int result = 1;
    if (run_stats == NULL) {
        return result;
    }
    AI_dictEntry *entry = AI_dictFind(run_stats, runkey);
    if (entry) {
        *rstats = AI_dictGetVal(entry);
        result = 0;
    }
    return result;
}

void RedisAI_FreeRunStats(RedisModuleCtx *ctx, struct RedisAI_RunStats *rstats) {
    RedisModule_FreeString(ctx, rstats->key);
    RAI_FreeRunStats(rstats);
}