#include "backends/util.h"

uint8_t backends_intra_op_parallelism;  //  number of threads used within an
//  individual op for parallelism.
uint8_t
    backends_inter_op_parallelism;  //  number of threads used for parallelism
                                    //  between independent operations.

int parseDeviceStr(const char* devicestr, RAI_Device* device, int64_t* deviceid) {
  if (strcasecmp(devicestr, "CPU") == 0) {
    *device = RAI_DEVICE_CPU;
    *deviceid = -1;
  }
  else if (strcasecmp(devicestr, "GPU") == 0) {
    *device = RAI_DEVICE_GPU;
    *deviceid = -1;
  }
  else if (strncasecmp(devicestr, "GPU:", 4) == 0) {
    *device = RAI_DEVICE_GPU;
    sscanf(devicestr, "GPU:%lld", deviceid);
  }
  else {
    return 0;
  }

  return 1;
}

/**
 *
 * @return number of threads used within an individual op for parallelism.
 */
uint8_t getBackendsInterOpParallelism(){
    return backends_inter_op_parallelism;
}

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsInterOpParallelism(uint8_t num_threads){
  int result = 1;
  if (num_threads>=0){
    backends_inter_op_parallelism=num_threads;
    result=0;
  }
  return result;
}


/**
 *
 * @return
 */
uint8_t getBackendsIntraOpParallelism(){
    return backends_intra_op_parallelism;
}

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsIntraOpParallelism(uint8_t num_threads){
  int result = 1;
  if (num_threads>=0){
    backends_intra_op_parallelism=num_threads;
    result=0;
  }
  return result;
}