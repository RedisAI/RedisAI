#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

typedef enum {
  RAI_BACKEND_TENSORFLOW = 0,
  RAI_BACKEND_TORCH,
  RAI_BACKEND_ONNXRUNTIME,
} RAI_Backend;

// NOTE: entry in queue hash is formed by
// device * MAX_DEVICE_ID + deviceid

typedef enum {
  RAI_DEVICE_CPU = 0,
  RAI_DEVICE_GPU = 1
} RAI_Device;

#define RAI_ENC_VER 400

//#define RAI_COPY_RUN_INPUT
#define RAI_COPY_RUN_OUTPUT
#define RAI_PRINT_BACKEND_ERRORS

// #define RAI_OVERRIDE_AOF_CHECK

#endif /* SRC_CONFIG_H_ */
