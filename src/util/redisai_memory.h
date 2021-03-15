/*
 * memory.h
 *
 *  Created on: Oct 18, 2018
 *      Author: meir
 */

#ifndef SRC_REDISAI_MEMORY_H_
#define SRC_REDISAI_MEMORY_H_

#include "redismodule.h"
#include "stdlib.h"
#include <string.h>

#ifdef VALGRIND
#define RA_ALLOC   malloc
#define RA_CALLOC  calloc
#define RA_REALLOC realloc
#define RA_FREE    free
#define RA_STRDUP  strdup
#else
#define RA_ALLOC   RedisModule_Alloc
#define RA_CALLOC  RedisModule_Calloc
#define RA_REALLOC RedisModule_Realloc
#define RA_FREE    RedisModule_Free
#define RA_STRDUP  RedisModule_Strdup
#endif

#endif /* SRC_REDISAI_MEMORY_H_ */
