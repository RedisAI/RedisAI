
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "../redisai_memory.h"
#include "redismodule.h"

#ifndef __QUEUE_H
#define __QUEUE_H

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
queueItem *queuePop(queue *queue);
queueItem *queueFront(queue *queue);
queueItem *queueNext(queueItem *item);
queueItem *queueEvict(queue *queue, queueItem *item);
long long queueLength(queue *queue);
void queueRelease(queue *queue);

typedef struct RunQueueInfo {
  pthread_mutex_t run_queue_mutex;
  pthread_cond_t queue_condition_var;
  queue *run_queue;
  pthread_t *threads;
} RunQueueInfo;

#endif /* __QUEUE_H */