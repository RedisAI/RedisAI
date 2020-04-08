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
* **type**: the tensor's data type can be one of: 'FLOAT', 'DOUBLE', 'INT8', 'INT16', 'INT32', 'INT64', 'UINT8' or 'UINT16'
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
* **TAG**: an optional string for tagging the model, such as a version number or any arbitrary identifier
* **BATCHSIZE**: batch incoming requests from multiple clients if they hit the same model and if input tensors have the same
                shape. Upon MODELRUN, the request queue is visited, input tensors from compatible requests are concatenated
                along the 0-th (batch) dimension, up until BATCHSIZE is exceeded. The model is then run for the entire batch,
                results are unpacked back among the individual requests and the respective clients are unblocked.
                If the batch size of the inputs to the first request in the queue exceeds BATCHSIZE, the request is served
                in any case. Default is 0 (no batching).
* **MINBATCHSIZE**: Do not execute a MODELRUN until the batch size has reached MINBATCHSIZE. This is primarily used to force
                   batching during testing, but it can also be used under normal operation. In this case, note that requests
                   for which MINBATCHSIZE is not reached will hang indefinitely.
                   Default is 0 (no minimum batch size).
* **INPUTS**: one or more names of the model's input nodes (applicable only for TensorFlow models)
* **OUTPUTS**: one or more names of the model's output nodes (applicable only for TensorFlow models)
* **model**: the Protobuf-serialized model

_Return_

A simple 'OK' string or an error.

**Examples**

This example shows to set a model 'mymodel' key using the contents of a local file with [`redis-cli`](https://redis.io/topics/cli). Refer to the [Clients Page](clients.md) for additional client choice that are native to your programming language:

```
$ cat mobilenet_v2_1.4_224_frozen.pb | redis-cli -x AI.MODELSET mymodel TF CPU INPUTS input OUTPUTS MobilenetV2/Predictions/Reshape_1
OK
```

```sql
AI.MODELSET mnist_net ONNX CPU TAG mnist:lenet:v0.1 < mnist.onnx
```

```sql
AI.MODELSET mnist_net ONNX CPU BATCHSIZE 10 < mnist.onnx
```

```sql
AI.MODELSET resnet18 TF CPU BATCHSIZE 10 MINBATCHSIZE 6 INPUTS in1 OUTPUTS linear4 < foo.pb
```

## AI.MODELGET
The **`AI.MODELGET`** command returns a model's meta data and blob stored as a key's value.

**Redis API**

```
AI.MODELGET <key> [META | BLOB]
```

_Arguments

* **key**: the model's key name
* META - Only return information on backend, device and tag
* BLOB - Return information on backend, device and tag, as well as a binary blob containing the serialized model

_Return_

A Bulk String that is the Protobuf-serialized representation of the model by the backend.

The command returns a list of key-value strings, namely `BACKEND backend DEVICE device TAG tag [BLOB blob]`.

**Examples**

Assuming that your model is stored under the 'mymodel' key, you can use the following read and save it to local file 'model.ext' with [`redis-cli`](https://redis.io/topics/cli):

```
$ redis-cli --raw AI.MODELGET mymodel > model.ext
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

NOTE: `_MODELLIST` is EXPERIMENTAL and might be removed in future versions.

List all models. Returns a list of (key, tag) pairs.

```sql
AI._MODELLIST
```

### _MODELLIST Example

```sql
AI.MODELSET model1 TORCH GPU TAG resnet18:v1 < foo.pt
AI.MODELSET model2 TORCH GPU TAG resnet18:v2 < bar.pt

AI._MODELLIST

>  1) 1) "model1"
>  1) 2) "resnet18:v1"
>  2) 1) "model2"
>  2) 2) "resnet18:v2"
```

## AI.SCRIPTSET
The **`AI.SCRIPTSET`** command stores a [TorchScript](https://pytorch.org/docs/stable/jit.html) as the value of a key.

**Redis API**

```
AI.SCRIPTSET <key> <device> [TAG tag] "<script>"
```

_Arguments_


* **key**: the script's key name
* TAG tag - Optional string tagging the script, such as a version number or other identifier
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
$ cat addtwo.py | redis-cli -x AI.SCRIPTSET myscript CPU
OK
```

```sql
AI.SCRIPTSET addscript GPU TAG myscript:v0.1 < addtwo.txt
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

Array consisting of two items:
!!!!The command returns a list of key-value strings, namely `DEVICE device TAG tag [SOURCE source]`.

1. The script's device as a String
2. The script's source code as a String

**Examples**

The following shows how to read the script stored at the 'myscript' key:

```
redis> AI.SCRIPTGET myscript
1) CPU
2) def addtwo(a, b):
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

NOTE: `_SCRIPTLIST` is EXPERIMENTAL and might be removed in future versions.

List all scripts. Returns a list of (key, tag) pairs.

```sql
AI._SCRIPTLIST
```

### _SCRIPTLIST Example

```sql
AI.SCRIPTSET script1 GPU TAG addtwo:v0.1 < addtwo.txt
AI.SCRIPTSET script2 GPU TAG addtwo:v0.2 < addtwo.txt

AI._SCRIPTLIST

>  1) 1) "script1"
>  1) 2) "addtwo:v0.1"
>  2) 1) "script2"
>  2) 2) "addtwo:v0.2"
```
---

## AI.INFO
The **`AI.INFO`** command returns information about the execution a model or a script.

Runtime information is collected each time that [`AI.MODELRUN`](#aimodelrun) or [`AI.SCRIPTRUN`]|(#aiscriptrun) is called. The information is stored locally by the executing RedisAI engine, so when deployed in a cluster each shard stores its own runtime information.

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

```sql
redis> AI.INFO myscript RESETSTAT
OK
```
