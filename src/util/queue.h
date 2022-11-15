/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "util/redisai_memory.h"
#include "redismodule.h"

typedef struct queueItem {
    struct queueItem *next;
    struct queueItem *prev;
    void *value;
} queueItem;

typedef struct queue {
    queueItem *front;
    queueItem *back;
    void (*free)(void *ptr);
    unsigned long len;
} queue;

queue *queueCreate(void);
void queuePush(queue *queue, void *value);
void queuePushFront(queue *queue, void *value);
queueItem *queuePop(queue *queue);
queueItem *queueFront(queue *queue);
queueItem *queueNext(queueItem *item);
queueItem *queueEvict(queue *queue, queueItem *item);
unsigned long queueLength(queue *queue);
void queueRelease(queue *queue);
