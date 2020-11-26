#ifndef SRC_BACKENDS_UTIL_H_
#define SRC_BACKENDS_UTIL_H_

#include <stdint.h>
#include <stdio.h>
#include <strings.h>

#include "config.h"

int parseDeviceStr(const char *devicestr, RAI_Device *device, int64_t *deviceid);

#endif /* SRC_BACKENDS_UTIL_H_ */
