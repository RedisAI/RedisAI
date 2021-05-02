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
