#pragma once

#include "background_workers.h"

RunQueueInfo *modelSet_QueueInfo;

// We use this generic struct to enable future extension - This BG workers may
// have more purposes (monitoring, statistics etc...)
typedef struct Job {
    void (*Execute)(void *);
    void *args;
} Job;

typedef struct ModelSetCtx {
    RedisModuleString **args;
    RedisModuleBlockedClient *client;
    RAI_Model *model;
    RAI_Error *err;
} ModelSetCtx;

Job *JobCreate(void *args, void (*CallBack)(void *));

void JobFree(Job *job);

int Init_BG_ModelSet();

void ModelSet_FreeData(RedisModuleCtx *ctx, void *err);

int RedisAI_ModelSet_Reply(RedisModuleCtx *ctx, RedisModuleString **argv, int argc);

void *RedisAI_ModelSet_ThreadMain(void *arg);

void ModelSet_Execute(void *args);
