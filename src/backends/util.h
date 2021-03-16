#pragma once

#include <stdint.h>
#include <stdio.h>
#include <strings.h>

#include "config/config.h"

int parseDeviceStr(const char *devicestr, RAI_Device *device, int64_t *deviceid);
