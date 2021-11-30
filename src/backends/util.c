#include <stdlib.h>
#include <errno.h>
#include "backends/util.h"

int parseDeviceStr(const char *device_str, RAI_Device *device, int64_t *device_id) {
    if (strncasecmp(device_str, "CPU", 3) == 0) {
        *device = RAI_DEVICE_CPU;
        *device_id = -1;
    } else if (strcasecmp(device_str, "GPU") == 0) {
        *device = RAI_DEVICE_GPU;
        *device_id = -1;
    } else if (strncasecmp(device_str, "GPU:", 4) == 0) {
        *device = RAI_DEVICE_GPU;
        // Convert the id string into a number, returns zero if no valid conversion could be
        // preformed, and sets errno in case of overflow.
        long long id = strtoll(device_str + 4, NULL, 0);
        if (errno == ERANGE)
            return 0;
        *device_id = id;
    } else {
        return 0;
    }
    return 1;
}
