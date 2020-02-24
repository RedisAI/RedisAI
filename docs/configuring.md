# Configuration

RedisAI supports both run-time configuration options and others that should be specified when loading the module. 

## Configuration Options During Loading

In general, passing configuration options is done by appending arguments after the `--loadmodule` argument in the command line, `loadmodule` configuration directive in a Redis config file, or the `MODULE LOAD` command. 

The module dynamic library `redisai.so` can be located in any path, provided that we specify the full path. The additional arguments are options passed to the module. Currently the supported options are:

- `BACKENDSPATH`: specify the default backends path used when loading a dynamic backend library.
- `TORCH`: specify the location of the PyTorch backend library, and dynamically load it. The location can be given in two ways, absolute or relative to the `<BACKENDSPATH>`. Using this option replaces the need for loading the PyTorch backend on runtime.
- `TF`: specify the location of the TensorFlow backend library, and dynamically load it. The location can be given in two ways, absolute or relative to the `<BACKENDSPATH>`. Using this option replaces the need for loading the TensorFlow backend on runtime.
- `TFLITE`: specify the location of the TensorFlow Lite backend library, and dynamically load it. The location can be given in two ways, absolute or relative to the `<BACKENDSPATH>`. Using this option replaces the need for loading the TensorFlow Lite backend on runtime.
- `ONNX`: specify the location of the ONNXRuntime backend library, and dynamically load it. The location can be given in two ways, absolute or relative to the `<BACKENDSPATH>`. Using this option replaces the need for loading the ONNXRuntime backend on runtime.
- `THREADS_PER_QUEUE`: specify the fixed number of worker threads up front per device. This option is described in detail at [THREADS_PER_QUEUE](##THREADS_PER_QUEUE) section and can be only set when loading the module.


### Configuration Examples

In redis.conf:

```
loadmodule redisai.so OPT1 OPT2
```

From redis-cli:

```
127.0.0.6379> MODULE load redisai.so OPT1 OPT2
```

From command line using relative path:

```
$ redis-server --loadmodule ./redisai.so OPT1 OPT2
```

From command line using full path:

```
$ redis-server --loadmodule /usr/lib/redis/modules/redisai.so OPT1 OPT2
```


### THREADS_PER_QUEUE

```
THREADS_PER_QUEUE {number}
```
Enable configuring the main thread to create a fixed number of worker threads up front per device. This controls the maximum number of threads to be used for parallel execution of independent different operations. 

This option can significantly improve the model run performance for simple models (models that require low computation effort), since there is usually room for extra computation on modern CPU's and hardware accelerators (GPUs, TPUs, etc.).

#### THREADS_PER_QUEUE Default

By default only one worker thread is used per device. 

#### THREADS_PER_QUEUE Example

```
$ redis-server --loadmodule ./redisai.so THREADS_PER_QUEUE 4
```

---


## Setting Configuration Options In Run-Time

### AI.CONFIG BACKENDSPATH

Specify the default backends path to use when dynamically loading a backend. 

```sql
AI.CONFIG BACKENDSPATH <default_location_of_backend_libraries>
```

#### AI.CONFIG BACKENDSPATH Example


```sql
AI.CONFIG BACKENDSPATH /usr/lib/redis/modules/redisai/backends
```

### AI.CONFIG LOADBACKEND

Load a DL/ML backend.

```sql
AI.CONFIG LOADBACKEND <backend_identifier> <location_of_backend_library>
```

RedisAI currently supports PyTorch (libtorch), Tensorflow (libtensorflow), TensorFlow Lite, and ONNXRuntime as backends. 

Allowed backend identifiers are:
-  `TF` (TensorFlow)
-  `TFLITE` (TensorFlow Lite)
-  `TORCH` (PyTorch)
-  `ONNX` (ONNXRuntime)



By default, RedisAI starts with the ability to set and get tensor data, but setting and running models and scritps requires a computing backend to be loaded, which can be done during loading, as [explained above](##-Configuration-Options-During-Loading), or at or run-time using the `AI.CONFIG` commmand.

This command allows to dynamically load a backend by specifying the backend identifier and the path to the backend library. Currently, once loaded, a backend cannot be unloaded, and there can be at most one backend per identifier loaded.


if you don't specify a backend on load time, RedisAI will look into the default location lazily, when a model of a given backend is loaded.

The default location is the `<BACKENDSPATH>/backends` directory.  The location of the backend library can be given in two ways, absolute or relative.
If relative, it is relative to `<BACKENDSPATH>`.
From there, RedisAI will look for:
- ONNXRuntime dynamic library at: `redisai_onnxruntime/redisai_onnxruntime.so`
- TensorFlow dynamic library at: `redisai_tensorflow/redisai_tensorflow.so`
- TensorFlow Lite dynamic library at: `redisai_tflite/redisai_tflite.so`
- PyTorch dynamic library at: `redisai_torch/redisai_torch.so`

Any library dependency will be resolved automatically, and the mentioned directories are portable on all platforms.


#### AI.CONFIG LOADBACKEND Examples

 Load the TORCH backend, relative to `BACKENDSPATH`

```sql
AI.CONFIG LOADBACKEND TORCH redisai_torch/redisai_torch.so
```

 Load the TORCH backend, specifying full path


```sql
AI.CONFIG LOADBACKEND TORCH /usr/lib/redis/modules/redisai/backends/redisai_torch/redisai_torch.so
```
