#pragma once

#define REDISAI_VERSION_MAJOR 1
#define REDISAI_VERSION_MINOR 2
#define REDISAI_VERSION_PATCH 4

#define REDISAI_MODULE_VERSION                                                                     \
    (REDISAI_VERSION_MAJOR * 10000 + REDISAI_VERSION_MINOR * 100 + REDISAI_VERSION_PATCH)

/* API versions. */
#define REDISAI_LLAPI_VERSION 1

static const long long REDISAI_ENC_VER = 4;
