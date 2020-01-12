# Configuration

RedisAI supports both run-time configuration options and others that should be specified when loading the module. 

## Configuration Options Only Available During Loading

In general, passing configuration options is done by appending arguments after the `--loadmodule` argument in the command line, `loadmodule` configuration directive in a Redis config file, or the `MODULE LOAD` command. For example:

In redis.conf:

```
loadmodule redisai.so OPT1 OPT2
```

From redis-cli:

```
127.0.0.6379> MODULE load redisai.so OPT1 OPT2
```

From command line:

```
$ redis-server --loadmodule ./redisai.so OPT1 OPT2
```


## THREADS_PER_QUEUE {number}

Enable configuring the main thread to create a fixed number of worker threads up front per device. This controls the maximum number of threads to be used for parallel execution of independent different operations. 


This option can significantly improve the model run performance for simple models.


### Default

By default only one worker thread is used per device. 

### Example

```
$ redis-server --loadmodule ./redisai.so THREADS_PER_QUEUE 4
```

---


## Setting Configuration Options In Run-Time

## AI.CONFIG LOADBACKEND

Enables setting run-time configuration options.

### AI.CONFIG LOADBACKEND Example

Load a DL/ML backend.

By default, RedisAI starts with the ability to set and get tensor data, but setting and running models and scritps requires a computing backend to be loaded. This command allows to dynamically load a backend by specifying the backend identifier and the path to the backend library. Currently, once loaded, a backend cannot be unloaded, and there can be at most one backend per identifier loaded.

```sql
AI.CONFIG LOADBACKEND <backend_identifier> <location_of_backend_library>
```

* allowed backend identifiers are: TF (TensorFlow), TORCH (PyTorch), ONNX (ONNXRuntime).

It is possible to specify backends at the command-line when starting `redis-server`, see example below.

> Load the TORCH backend

```sql
AI.CONFIG LOADBACKEND TORCH install/backend/redisai_torch/redisai_torch.so
```

> Load the TORCH backend at the command-line

```bash
redis-server --loadmodule install/redisai.so TORCH install/backend/redisai_torch/redisai_torch.so
```

This replaces the need for loading a backend using AI.CONFIG LOADBACKEND
