/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#define REDISAI_VERSION_MAJOR 99
#define REDISAI_VERSION_MINOR 99
#define REDISAI_VERSION_PATCH 99

#define REDISAI_MODULE_VERSION                                                                     \
    (REDISAI_VERSION_MAJOR * 10000 + REDISAI_VERSION_MINOR * 100 + REDISAI_VERSION_PATCH)

/* API versions. */
#define REDISAI_LLAPI_VERSION 1

static const long long REDISAI_ENC_VER = 4;
