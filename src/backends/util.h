/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <strings.h>

#include "config/config.h"

int parseDeviceStr(const char *device_str, RAI_Device *device, int64_t *device_id);
