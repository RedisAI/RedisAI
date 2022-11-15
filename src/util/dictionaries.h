/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "dict.h"

/**
 * @brief Dictionary key type: const char*. value type: void*.
 *
 */
extern AI_dictType AI_dictTypeHeapStrings;

/**
 * @brief Dictionary key type: RedisModuleString*. value type: void*.
 *
 */
extern AI_dictType AI_dictTypeHeapRStrings;

/**
 * @brief Dictionary key type: const char*. value type: arr.
 *
 */
extern AI_dictType AI_dictType_String_ArrSimple;
