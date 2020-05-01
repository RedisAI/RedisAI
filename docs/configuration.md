# RedisAI Configuration
RedisAI provides configuration options that control its operation. These options can be set when the module is bootstrapped, that is loaded to Redis, and in some cases also during runtime.

The following sections describe the configuration options the means for setting them.

## Setting Configuration Options

**Bootstrap Configuration**

Configuration options can be set when the module is loaded. The options are passed as a list of option names and their respective values. Configuration is supported both when using the `loadmodule` configuration directive as well as via the [Redis `MODULE LOAD` command](https://redis.io/commands/module-load).

!!! example "Example: Setting configuration options"
    For setting the module's configuration options from the command line use:

    ```
    redis-server --loadmodule /usr/lib/redis/modules/redisai.so <opt1> <val1> ...
    ```

    For setting the module's configuration options in with .conf file use the following format:

    ```
    loadmodule /usr/lib/redis/modules/redisai.so <opt1> <val1> ...
    ```

    For setting the module's configuration with the [`MODULE LOAD`](https://redis.io/commands/module-load) command use:

    ```
    redis> MODULE LOAD /usr/lib/redis/modules/redisai.so <opt1> <val1> ...
    ```

**Runtime Configuration**

Some configuration options may be set at runtime via the [`AI.CONFIG` command](commands.md#aiconfig).

Refer to each option's description for its runtime configurability.

## Backend
By default RedisAI doesn't load any of its backend libraries when it is initialized. Backend libraries are then loaded lazily when models that require them are loaded.

The following backend configuration options make it possible to have RedisAI preemptively load one or more backend libraries.

### BACKENDSPATH
The **BACKENDSPATH** configuration option sets the default base path that RedisAI will use for dynamically loading backend libraries.

_Expected Value_

A String that is an absolute path.

_Default Value_

This option is initialized with the module's path suffixed by '/backends'.

For example, if the module is loaded from '/usr/lib/redis/modules/redisai.so', this option's value will be set by default to "/usr/lib/redis/modules/backends".

_Runtime Configurability_

Supported.

**Examples**

To set the default backends path to '/usr/lib/backends' when loading the module from the command line use the following:

```sh
redis-server --loadmodule /usr/lib/redis/modules/redisai.so \
               BACKENDSPATH /usr/lib/backends
```


### INTER_OP_PARALLELISM
The **INTER_OP_PARALLELISM** configuration option sets the number of threads used for parallelism between independent operations, by backend libraries. By default, 0 means RedisAI will not enforce a configuration and use the default configuration for each backend library.

_Expected Value_

An Integer equal or greater than zero.

_Default Value_

0

_Runtime Configurability_

Not supported.

**Examples**

To set the number of threads used for parallelism between independent operations to 1, when loading the module from the command line use the following:

```sh
redis-server --loadmodule /usr/lib/redis/modules/redisai.so \
               INTER_OP_PARALLELISM 1
```

### INTRA_OP_PARALLELISM
The **INTRA_OP_PARALLELISM** configuration option sets the number of threads used within an individual operation, by backend libraries. By default, 0 means RedisAI will not enforce a configuration and use the default configuration for each backend library.

_Expected Value_

An Integer greater or equal than zero.

_Default Value_

0

_Runtime Configurability_

Not supported.

**Examples**

To set the number of threads used within an individual operation to 1, when loading the module from command line use the following:

```sh
redis-server --loadmodule /usr/lib/redis/modules/redisai.so \
               INTRA_OP_PARALLELISM 1
```

### TF, TFLITE, TORCH and ONNX
The **TF**, **TFLITE**, **TORCH** and **ONNX** configuration options load the TensorFlow, TensorFlow Lite, PyTorch and ONNXRuntime backend libraries, respectively.

Each of the options requires a path to the library's binary. If the provided path is a relative path, it is resolved by prefixing the `BACKENDSPATH` to it. If the provided path is an absolute path then it is used as is.

_Expected Value_

A String that is a path.

_Default Value_

Each of the options has its own default values as follows:

* **TF**: `"<BACKENDSPATH>/redisai_tensorflow/redisai_tensorflow.so"`
* **TFLITE**: `"<BACKENDSPATH>/redisai_tflite/redisai_tflite.so"`
* **TORCH**: `"<BACKENDSPATH>/redisai_torch/redisai_torch.so"`
* **ONNX**: `"<BACKENDSPATH>/redisai_onnxruntime/redisai_onnxruntime.so"`

_Runtime Configurability_

Supported.

**Examples**

This loads the PyTorch backend at boot time located at '/usr/lib/backends/torch_custom/torch_xyz.so' using a relative path and `BACKENDSPATH`:

```
redis-server --loadmodule /usr/lib/redis/modules/redisai.so \
               BACKENDSPATH /usr/lib/backends \
               TORCH torch_custom/torch_xyz.so
```

## Device

### THREADS_PER_QUEUE
The **THREADS_PER_QUEUE** configuration option controls the number of worker threads allocated to each device's job queue. Multiple threads can be used for executing different independent operations in parallel.

This option significantly improves the performance of simple, low-effort computation-wise models since there is spare computation cycle available from modern CPUs and hardware accelerators (GPUs, TPUs, ...).

_Expected Value_

An Integer greater than zero.

_Default Value_

1

_Runtime Configurability_

Not supported.

**Examples**

To set the number of threads per queue to 4 when loading the module from the command line use the following:

```
redis-server --loadmodule /usr/lib/redis/modules/redisai.so \
               THREADS_PER_QUEUE 4
```
