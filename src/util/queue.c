#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "queue.h"
#include "../redisai_memory.h"
#include "redismodule.h"

queue *queueCreate(void) {
  struct queue *queue;

  if ((queue = RedisModule_Calloc(1, sizeof(*queue))) == NULL) return NULL;

  queue->front = queue->back = NULL;
  queue->len = 0;
  queue->free = NULL;
  return queue;
}

void queuePush(queue *queue, void *value) {
  queueItem *item;

  if ((item = RedisModule_Calloc(1, sizeof(*item))) == NULL) return;
  item->value = value;
  item->next = NULL;
  item->prev = NULL;

  if (queue->len == 0) {
    queue->front = queue->back = item;
  } else {
    queue->back->next = item;
    item->prev = queue->back;
    queue->back = item;
  }
  queue->len++;
}

void queueUnpop(queue *queue, void *value) {
  queueItem *item;

  if ((item = RedisModule_Calloc(1, sizeof(*item))) == NULL) return;
  item->value = value;
  item->next = NULL;
  item->prev = NULL;

  if (queue->len == 0) {
    queue->front = queue->back = item;
  } else {
    queue->front->prev = item;
    item->next = queue->front;
    queue->front = item;
  }
  queue->len++;
}

queueItem *queuePop(queue *queue) {
  queueItem *item = queue->front;
  if (item == NULL) {
    return NULL;
  }
  queue->front = item->next;
  if (queue->front != NULL) {
    queue->front->prev = NULL;
  }
  if (item == queue->back) {
    queue->back = NULL;
  }
  item->next = NULL;
  item->prev = NULL;
  queue->len--;
  return item;
}

queueItem *queueFront(queue *queue) { return queue->front; }

queueItem *queueNext(queueItem *item) { return item->next; }

queueItem *queueEvict(queue *queue, queueItem *item) {
  if (item == queue->front) {
    return queuePop(queue);
  } else if (item == queue->back) {
    queue->back = item->prev;
    queue->back->next = NULL;
  } else {
    item->prev->next = item->next;
    item->next->prev = item->prev;
  }

  item->next = NULL;
  item->prev = NULL;
  queue->len--;
  return item;
}

long long queueLength(queue *queue) { return queue->len; }

void queueRelease(queue *queue) {
  unsigned long len;
  queueItem *current;

  len = queue->len;
  while (len--) {
    current = queuePop(queue);
    if (current && queue->free) queue->free(current->value);
    RedisModule_Free(current);
  }
  queue->front = queue->back = NULL;
  queue->len = 0;
}
