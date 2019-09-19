#ifndef SRC_BACKENDS_UTIL_H_
#define SRC_BACKENDS_UTIL_H_

#include "config.h"
#include <stdint.h>
#include <stdio.h>
#include <strings.h>

int parseDeviceStr(const char* devicestr, RAI_Device* device, int64_t* deviceid);

#endif /* SRC_BACKENDS_UTIL_H_ */
