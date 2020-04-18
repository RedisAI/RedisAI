# RedisAI Commands
RedisAI is a Redis module, and as such it implements several data types and the respective commands to use them.

All of RedisAI's commands are begin with the `AI.` prefix. The following sections describe these commands.

**Syntax Conventions**

The following conventions are used for describing the RedisAI Redis API:

* `COMMAND`: a command or an argument name
* `<mandatory>`: a mandatory argument
* `[optional]`: an optional argument
* `"`: a literal double quote character
* `|`: an exclusive logical or operator
* `...`: more of the same as before

## AI.TENSORSET
The **`AI.TENSORSET`** command stores a tensor as the value of a key.

**Redis API**

```
AI.TENSORSET <key> <type>
   <shape> [shape ...] [BLOB <data> | VALUES <val> [val ...]]
```

_Arguments_

* **key**: the tensor's key name
* **type**: the tensor's data type can be one of: `FLOAT`, `DOUBLE`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8` or `UINT16`
* **shape**: one or more dimensions, or the number of elements per axis, for the tensor
* **BLOB**: indicates that data is in binary format and is provided via the subsequent `data` argument
* **VALUES**: indicates that data is numeric and is provided by one or more subsequent `val` arguments

_Return_

A simple 'OK' string or an error.

**Examples**

Given the following: $\begin{equation*} A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \end{equation*}$

This will set the key 'mytensor' to the 2x2 RedisAI tensor:

```
redis> AI.TENSORSET mytensor FLOAT 2 2 VALUES 1 2 3 4
OK
```

!!! note "Uninitialized Tensor Values"
    As both `BLOB` and `VALUES` are optional arguments, it is possible to use the `AI.TENSORSET` to create an uninitialized tensor.
    TBD: why is that useful?

!!! important "Using `BLOB` is preferable to `VALUES`"
    While it is possible to set the tensor using binary data or numerical values, it is recommended that you use the `BLOB` option. It requires less resources and performs better compared to specifying the values discretely.

## AI.TENSORGET
The **`AI.TENSORGET`** command returns a tensor stored as key's value.

**Redis API**

```
AI.TENSORGET <key> <format>
```

_Arguments_

* **key**: the tensor's key name
* **format**: the reply's format can be on of the following:
    * **BLOB**: returns the binary representation of the tensor's data
    * **VALUES**: returns the numerical representation of the tensor's data
    * **META**: returns the tensor's meta data exclusively

_Return_

Array containing the tensor's data.

The returned array consists of the following elements:

1. The tensor's data type as a String
1. The tensor's shape as an Array consisting of an item per dimension
1. The tensor's data when called with the `BLOB` or `VALUES` arguments

The reply's third element type, that is the tensor's data, depends on the given argument:

* **BLOB**: the tensor's binary data as a String
* **VALUES**: the tensor's values an Array
* **META**: when used no data is returned

**Examples**

Given a tensor value stored at the 'mytensor' key:

```
redis> AI.TENSORSET mytensor FLOAT 2 2 VALUES 1 2 3 4
OK
```

The following shows how the tensor's binary data is read:

```
redis> AI.TENSORGET mytensor BLOB
1) FLOAT
2) 1) (integer) 2
   2) (integer) 2
3) "\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
```

!!! important "Using `BLOB` is preferable to `VALUES`"
    While it is possible to get the tensor as binary data or numerical values, it is recommended that you use the `BLOB` option. It requires less resources and performs better compared to returning the values discretely.

## AI.MODELSET
The **`AI.MODELSET`** commands stores a model as the value of a key.

**Redis API**

```
AI.MODELSET <key> <backend> <device>
    [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]]
    [INPUTS <name> ...] [OUTPUTS name ...] <model>
```

_Arguments_

* **key**: the model's key name
* **backend**: the backend for the model can be one of:
    * **TF**: a TensorFlow backend
    * **TORCH**: a PyTorch backend
    * **ONNX**: a ONNX backend
* **device**: the device that will execute the model can be of:
    * **CPU**: a CPU device
    * **GPU**: a GPU device
* **TAG**: an optional string for tagging the model such as a version number or any arbitrary identifier
* **BATCHSIZE**: when provided with an `n` that is greater than 0, the engine will batch incoming requests from multiple clients that use the model with input tensors of the same shape. When `AI.MODELRUN` is called the requests queue is visited and input tensors from compatible requests are concatenated along the 0th (batch) dimension until `n` is exceeded. The model is then run for the entire batch and the results are unpacked back to the individual requests unblocking their respective clients. If the batch size of the inputs to of first request in the queue exceeds `BATCHSIZE`, the request is served immediately (default value: 0).
* **MINBATCHSIZE**: when provided with an `m` that is greater than 0, the engine will postpone calls to `AI.MODELRUN` until the batch's size had reached `m`. This is primarily used to force batching during testing, but it can also be used under normal operation. In this case, note that requests for which `m` is not reached will hang indefinitely (defai;t value: 0).
* **INPUTS**: one or more names of the model's input nodes (applicable only for TensorFlow models)
* **OUTPUTS**: one or more names of the model's output nodes (applicable only for TensorFlow models)
* **model**: the Protobuf-serialized model

_Return_

A simple 'OK' string or an error.

**Examples**

This example shows to set a model 'mymodel' key using the contents of a local file with [`redis-cli`](https://redis.io/topics/cli). Refer to the [Clients Page](clients.md) for additional client choice that are native to your programming language:

```
$ cat resnet50.pb | redis-cli -x AI.MODELSET mymodel TF CPU TAG imagenet:5.0 INPUTS images OUTPUTS output
OK
```

## AI.MODELGET
The **`AI.MODELGET`** command returns a model's meta data and blob stored as a key's value.

**Redis API**

```
AI.MODELGET <key> [META | BLOB]
```

_Arguments

* **key**: the model's key name
* **META**: will only return the model's meta information (default)
* **BLOB**: will return the model's meta information and a blob containing the serialized model

_Return_

An array of alternating key-value pairs as follows:

1. **BACKEND**: the backend used by the model as a String
1. **DEVICE**: the device used to execute the model as a String
1. **TAG**: the model's tag as a String
1. **BLOB**: a blob containing the serialized model (when called with the `BLOB` argument) as a String

**Examples**

Assuming that your model is stored under the 'mymodel' key, you can obtain its metadata with:

```
redis> 1) BACKEND
2) TF
3) DEVICE
4) CPU
5) TAG
6) imagenet:5.0
```

You can also save it to the local file 'model.ext' with [`redis-cli`](https://redis.io/topics/cli) like so:

```
$ redis-cli --raw AI.MODELGET mymodel BLOB > model.ext
```

## AI.MODELDEL
The **`AI.MODELDEL`** deletes a model stored as a key's value.

**Redis API**

```
AI.MODELDEL <key>
```

_Arguments_

* **key**: the model's key name

_Return_

A simple 'OK' string or an error.

**Examples**

Assuming that your model is stored under the 'mymodel' key, you can delete it like this:

```
redis> AI.MODELDEL mymodel
OK
```

!!! note "The `AI.MODELDEL` vis a vis the `DEL` command"
    The `AI.MODELDEL` is equivalent to the [Redis `DEL` command](https://redis.io/commands/del) and should be used in its stead. This ensures compatibility with all deployment options (i.e., stand-alone vs. cluster, OSS vs. Enterprise).

## AI.MODELRUN
The **`AI.MODELRUN`** command runs a model stored as a key's value using its specified backend and device. It accepts one or more input tensors and store output tensors.

The run request is put in a queue and is executed asynchronously by a worker thread. The client that had issued the run request is blocked until the model's run is completed. When needed, tensors' data is automatically copied to the device prior to execution.

!!! warning "Intermediate memory overhead"
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

**Redis API**

```
AI.MODELRUN <key> INPUTS <input> [input ...] OUTPUTS <output> [output ...]
```

_Arguments_

* **key**: the model's key name
* **INPUTS**: denotes the beginning of the input tensors keys' list, followed by one or more key names
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by one or more key names

_Return_

A simple 'OK' string or an error.

**Examples**

Assuming that running the model that's stored at 'mymodel' with the tensor 'mytensor' as input outputs two tensors - 'classes' and 'predictions', the following command does that:

```
redis> AI.MODELRUN mymodel INPUTS mytensor OUTPUTS classes predictions
OK
```

## AI._MODELLIST
The **AI._MODELLIST** command returns all the models in the database.

!!! warning "Experimental API"
    `AI._MODELLIST` is an EXPERIMENTAL command that may be removed in future versions.

**Redis API**

```
AI._MODELLIST
```

_Arguments_

None.

_Return_

An array with an entry per model. Each entry is a an array with two entries:

1. The model's key name as a String
1. The model's tag as a String

**Examples**

```
redis> > AI._MODELLIST
1) 1) "mymodel"
   2) imagenet:5.0
```

## AI.SCRIPTSET
The **`AI.SCRIPTSET`** command stores a [TorchScript](https://pytorch.org/docs/stable/jit.html) as the value of a key.

**Redis API**

```
AI.SCRIPTSET <key> <device> [TAG tag] "<script>"
```

_Arguments_


* **key**: the script's key name
* **TAG**: an optional string for tagging the script such as a version number or any arbitrary identifier
* **device**: the device that will execute the model can be of:
    * **CPU**: a CPU device
    * **GPU**: a GPU device
* **script**: the script's source code

_Return_

A simple 'OK' string or an error.

**Examples**

Given the following contents of the file 'addtwo.py':

```python
def addtwo(a, b):
    return a + b
```

It can be stored as a RedisAI script using the CPU device with [`redis-cli`](https://redis.io/topics/rediscli) as follows:

```
$ cat addtwo.py | redis-cli -x AI.SCRIPTSET myscript addtwo CPU TAG myscript:v0.1
OK
```

## AI.SCRIPTGET
The **`AI.SCRIPTGET`** command returns the [TorchScript](https://pytorch.org/docs/stable/jit.html) stored as a key's value.

**Redis API**

Get script metadata and source.

```
AI.SCRIPTGET <key>
```

_Arguments_

* **key**: the script's key name

_Return_

An array with alternating entries that represent the following key-value pairs:
!!!!The command returns a list of key-value strings, namely `DEVICE device TAG tag [SOURCE source]`.

1. **DEVICE**: the script's device as a String
1. **TAG**: the scripts's tag as a String
1. **SOURCE**: the script's source code as a String

**Examples**

The following shows how to read the script stored at the 'myscript' key:

```
redis> AI.SCRIPTGET myscript
1) DEVICE
2) CPU
3) TAG
4) "myscript:v0.1"
5) SOURCE
6) def addtwo(a, b):
    return a + b
```

## AI.SCRIPTDEL
The **`AI.SCRIPTDEL`** deletes a script stored as a key's value.

**Redis API**

```
AI.SCRIPTDEL <key>
```

_Arguments_

* **key**: the script's key name

_Return_

A simple 'OK' string or an error.

**Examples**

```
redis> AI.SCRIPTDEL myscript
OK
```

!!! note "The `AI.SCRIPTDEL` vis a vis the `DEL` command"
    The `AI.SCRIPTDEL` is equivalent to the [Redis `DEL` command](https://redis.io/commands/del) and should be used in its stead. This ensures compatibility with all deployment options (i.e., stand-alone vs. cluster, OSS vs. Enterprise).


## AI.SCRIPTRUN
The **`AI.SCRIPTRUN`** command runs a script stored as a key's value on its specified device. It accepts one or more input tensors and store output tensors.

**Redis API**

```
AI.SCRIPTRUN <key> <function> INPUTS <input> [input ...] OUTPUTS <output> [output ...]
```

_Arguments_

* **key**: the script's key name
* **function**: the name of the function to run
* **INPUTS**: denotes the beginning of the input tensors keys' list, followed by one or more key names
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by one or more key names

_Return_

A simple 'OK' string or an error.

**Examples**

The following is an example of running the previously-created 'myscript' on two input tensors:

```
redis> AI.TENSORSET mytensor1 FLOAT 1 VALUES 40
OK
redis> AI.TENSORSET mytensor2 FLOAT 1 VALUES 2
OK
redis> AI.SCRIPTRUN myscript addtwo INPUTS mytensor1 mytensor2 OUTPUTS result
OK
redis> AI.TENSORGET result VALUES
1) FLOAT
2) 1) (integer) 1
3) 1) "42"
```

!!! warning "Intermediate memory overhead"
    The execution of scripts may generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

## AI._SCRIPTLIST
The **AI._SCRIPTLIST** command returns all the scripts in the database.

!!! warning "Experimental API"
    `AI._MODELLIST` is an EXPERIMENTAL command that may be removed in future versions.

**Redis API**

```
AI._SCRIPTLIST
```

_Arguments_

None.

_Return_

An array with an entry per script. Each entry is a an array with two entries:

1. The script's key name as a String
1. The script's tag as a String

**Examples**

```
redis> > AI._SCRIPTLIST
1) 1) "myscript"
   2) "myscript:v0.1"
```

## AI.INFO
The **`AI.INFO`** command returns information about the execution a model or a script.

Runtime information is collected each time that [`AI.MODELRUN`](#aimodelrun) or [`AI.SCRIPTRUN`](#aiscriptrun) is called. The information is stored locally by the executing RedisAI engine, so when deployed in a cluster each shard stores its own runtime information.

**Redis API**

```
AI.INFO <key> [RESETSTAT]
```

_Arguments_

* **key**: the key name of a model or script
* **RESETSTAT**: resets all statistics associated with the key

_Return_

An array with alternating entries that represent the following key-value pairs:

* **KEY**: a String of the name of the key storing the model or script value
* **TYPE**: a String of the type of value (i.e. 'MODEL' or 'SCRIPT')
* **BACKEND**: a String of the type of backend (always 'TORCH' for 'SCRIPT' value type)
* **DEVICE**: a String of the device where execution took place
* **DURATION**: the cumulative duration of executions in microseconds
* **SAMPLES**: the cumulative number of samples obtained from the 0th (batch) dimension (only applicable for RedisAI models)
* **CALLS**: the total number of executions
* **ERRORS**: the total number of errors generated by executions (excluding any errors generated during parsing commands)

When called with the `RESETSTAT` argument, the command returns a simple 'OK' string.

**Examples**

The following example obtains the previously-run 'myscript' script's runtime statistics:

```
redis> AI.INFO myscript
 1) KEY
 2) "myscript"
 3) TYPE
 4) SCRIPT
 5) BACKEND
 6) TORCH
 7) DEVICE
 8) CPU
 9) DURATION
10) (integer) 11391
11) SAMPLES
12) (integer) -1
13) CALLS
14) (integer) 1
15) ERRORS
16) (integer) 0
```

The runtime statistics for that script can be reset like so:

```
redis> AI.INFO myscript RESETSTAT
OK
```

## AI.CONFIG
The **AI.CONFIG** command sets the value of configuration directives at run-time, and allows loading DL/ML backends dynamically.

!!! info "Loading DL/ML Backends at Bootstrap"
    Instead of loading your backends dynamically, you can have RedisAI load them during bootstrap. See the [Configuration page](configuration.md) for more information.

**Redis API**
```
AI.CONFIG <BACKENDSPATH <path>> | <LOADBACKEND <backend> <path>>
```

_Arguments_

* **BACKENDSPATH**: Specifies the default base backends path to `path`. The backends path is used when dynamically loading a backend (default: '{module_path}/backends', where `module_path` is the module's path).
* **LOADBACKEND**: Loads the DL/ML backend specified by the `backend` identifier from `path`. If `path` is relative, it is resolved by prefixing the `BACKENDSPATH` to it. If `path` is absolute then it is used as is. The `backend` can be one of:
    * **TF**: the TensorFlow backend
    * **TFLITE**: The TensorFlow Lite backend
    * **TORCH**: The PyTorch backend
    * **ONNX**: ONNXRuntime backend

_Return_

A simple 'OK' string or an error.

**Examples**

The following sets the default backends path to '/usr/lib/redis/modules/redisai/backends':

```
redis> AI.CONFIG BACKENDSPATH /usr/lib/redis/modules/redisai/backends
OK
```

This loads the PyTorch backend with a path relative to `BACKENDSPATH`:

```
redis> AI.CONFIG LOADBACKEND TORCH redisai_torch/redisai_torch.so
OK
```

This loads the PyTorch backend with a full path:

```
redis> AI.CONFIG LOADBACKEND TORCH /usr/lib/redis/modules/redisai/backends/redisai_torch/redisai_torch.so
OK
```
