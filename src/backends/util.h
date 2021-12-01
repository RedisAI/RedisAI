#pragma once

#include <stdint.h>
#include <stdio.h>
#include <strings.h>

#include "config/config.h"

int parseDeviceStr(const char *device_str, RAI_Device *device, int64_t *device_id);
