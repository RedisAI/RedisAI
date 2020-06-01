#include "backends/util.h"

int parseDeviceStr(const char* devicestr, RAI_Device* device,
                   int64_t* deviceid) {
  // if (strcasecmp(devicestr, "CPU") == 0) {
  if (strncasecmp(devicestr, "CPU", 3) == 0) {
    *device = RAI_DEVICE_CPU;
    *deviceid = -1;
  } else if (strcasecmp(devicestr, "GPU") == 0) {
    *device = RAI_DEVICE_GPU;
    *deviceid = -1;
  } else if (strncasecmp(devicestr, "GPU:", 4) == 0) {
    *device = RAI_DEVICE_GPU;
    sscanf(devicestr, "GPU:%lld", deviceid);
  } else {
    return 0;
  }

  return 1;
}

