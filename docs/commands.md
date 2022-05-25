# RedisAI Commands
RedisAI is a Redis module, and as such it implements several data types and the respective commands to use them.

All of RedisAI's commands begin with the `AI.` prefix. The following sections describe these commands.

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
* **type**: the tensor's data type can be one of: `FLOAT`, `DOUBLE`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `BOOL` or `STRING`
* **shape**: one or more dimensions, or the number of elements per axis, for the tensor
* **BLOB**: indicates that data is in binary format and is provided via the subsequent `data` argument
* **VALUES**: indicates that data is given by values and is provided by one or more subsequent `val` arguments

_Return_

A simple 'OK' string or an error.

**Examples**

Given the following: $\begin{equation*} A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \end{equation*}$

This will set the key 'mytensor' to the 2x2 RedisAI tensor:

```
redis> AI.TENSORSET mytensor FLOAT 2 2 VALUES 1 2 3 4
OK
redis> AI.TENSORSET mytensor STRING 1 2 VALUES first second
OK
```

!!! note "Uninitialized Tensor Values"
    As both `BLOB` and `VALUES` are optional arguments, it is possible to use the `AI.TENSORSET` to create an uninitialized tensor (it will contain zeros at all entries).

!!! important "Using `BLOB` is preferable to `VALUES`"
    While it is possible to set the tensor using binary data or numerical values, it is recommended that you use the `BLOB` option. It requires fewer resources and performs better compared to specifying the values discretely.

###Boolean Tensors
The possible values for a tensor of type `BOOL` are `0` and `1`. The size of every bool element in a blob should be 1 byte.   

**Examples**

Here are two ways of creating the following boolean tensor: $\begin{equation*} A = \begin{bmatrix} 0 & 1 \\ 0 & 1 \\ \end{bmatrix} \end{equation*}$
```
redis> AI.TENSORSET my_bool_tensor BOOL 2 2 VALUES 0 1 0 1
OK
redis> AI.TENSORSET my_bool_tensor BOOL 2 2 BLOB "\x00\x01\x00\x01"
OK
```

###String Tensors
String tensors are tensors in which every element is a single utf-8 string (may or may not be null-terminated). A string element can be at any length, and it cannot contain another null character except for the last one if it is a null-terminated string.
A string tensor blob contains the encoded string elements concatenated, where the null character serves as a delimiter. Note that the size of string tensor blob equals to the total size of its elements, and it is not determined given the tensor's shapes (unlike in the rest of tensor types) 

**Examples**

Here are two ways of creating the same 2X2 string tensor:
```
redis> AI.TENSORSET my_str_tensor STRING 2 2 VALUES first second third fourth
OK
redis> AI.TENSORSET my_bool_tensor STRING 2 2 BLOB "first\x00second\x00third\x00fourth\x00"
OK
```

## AI.TENSORGET
The **`AI.TENSORGET`** command returns a tensor stored as key's value.

**Redis API**

```
AI.TENSORGET <key> [META] [format]
```

_Arguments_

* **key**: the tensor's key name
* **META**: returns the tensor's metadata
* **format**: the tensor's reply format can be one of the following:
    * **BLOB**: returns the binary representation of the tensor's data
    * **VALUES**: returns the actual values of the tensor's data

_Return_

Depending on the specified reply format:

 * **META**: Array containing the tensor's metadata exclusively. The returned array consists of the following elements:
    1. The tensor's data type as a String
    1. The tensor's shape as an Array consisting of an item per dimension
 * **BLOB**: the tensor's binary data as a String. If used together with the **META** option, the binary data string will put after the metadata in the array reply.
 * **VALUES**: Array containing the values of the tensor's data. If used together with the **META** option, the binary data string will put after the metadata in the array reply.
* Default: **META** and **BLOB** are returned by default, in case that none of the arguments above is specified. 


**Examples**

Given tensor values stored at 'my_tensor' and _my_str_tensor keys:

```
redis> AI.TENSORSET my_tensor FLOAT 2 2 VALUES 1 2 3 4
OK
redis> AI.TENSORSET my_str_tensor STRING 2 2 VALUES first second third fourth
OK
```

The following shows how to retrieve the tensors' metadata:

```
redis> AI.TENSORGET my_tensor META
1) "dtype"
2) "FLOAT"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2

redis> AI.TENSORGET my_str_tensor META
1) "dtype"
2) "STRING"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2
```

The following shows how to retrieve the tensors' values as an Array:

```
redis> AI.TENSORGET my_tensor VALUES
1) "1"
2) "2"
3) "3"
4) "4"

redis> AI.TENSORGET my_str_tensor VALUES
1) "first"
2) "second"
3) "third"
4) "fourth"
```

The following shows how to retrieve the tensors' binary data as a String:

```
redis> AI.TENSORGET my_tensor BLOB
"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"

redis> AI.TENSORGET my_str_tensor BLOB
"first\x00second\x00third\x00fourth\x00"
```


The following shows how the combine the retrieval of the tensors' metadata, and the tensors' values as an Array:

```
redis> AI.TENSORGET my_tensor META VALUES
1) "dtype"
2) "FLOAT"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2
5) "values"
6) 1) "1"
   2) "2"
   3) "3"
   4) "4"

redis> AI.TENSORGET my_str_tensor META VALUES
1) "dtype"
2) "STRING"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2
5) "values"
6) 1) "first"
   2) "second"
   3) "third"
   4) "fourth"
```

The following shows how the combine the retrieval of the tensors' metadata, and binary data as a String:

```
redis> AI.TENSORGET my_tensor META BLOB
1) "dtype"
2) "FLOAT"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2
5) "blob"
6) "\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"

redis> AI.TENSORGET my_str_tensor META BLOB
1) "dtype"
2) "STRING"
3) "shape"
4) 1) (integer) 2
   2) (integer) 2
5) "blob"
6) "first\x00second\x00third\x00fourth\x00"
```

!!! important "Using `BLOB` is preferable to `VALUES`"
    While it is possible to get the tensor as binary data or by values, it is recommended that you use the `BLOB` option. It requires fewer resources and performs better compared to returning the values discretely.

## AI.MODELSTORE
The **`AI.MODELSTORE`** command stores a model as the value of a key.

**Redis API**

```
AI.MODELSTORE <key> <backend> <device>
    [TAG <tag>] [BATCHSIZE <n> [MINBATCHSIZE <m> [MINBATCHTIMEOUT <t>]]]
    [INPUTS <input_count> <name> ...] [OUTPUTS <output_count> <name> ...] BLOB <model>
```

_Arguments_

* **key**: the model's key name
* **backend**: the backend for the model can be one of:
    * **TF**: a TensorFlow backend
    * **TFLITE**: The TensorFlow Lite backend
    * **TORCH**: a PyTorch backend
    * **ONNX**: a ONNX backend
* **device**: the device that will execute the model can be of:
    * **CPU**: a CPU device
    * **GPU**: a GPU device
    * **GPU:0**, ..., **GPU:n**: a specific GPU device on a multi-GPU system
* **TAG**: an optional string for tagging the model such as a version number or any arbitrary identifier
* **BATCHSIZE**: when provided with an `n` that is greater than 0, the engine will batch incoming requests from multiple clients that use the model with input tensors of the same shape. When `AI.MODELEXECUTE` (or `AI.MODELRUN`) is called the requests queue is visited and input tensors from compatible requests are concatenated along the 0th (batch) dimension until `n` is exceeded. The model is then run for the entire batch and the results are unpacked back to the individual requests unblocking their respective clients. If the batch size of the inputs to of first request in the queue exceeds `BATCHSIZE`, the request is served immediately (default value: 0).
* **MINBATCHSIZE**: when provided with an `m` that is greater than 0, the engine will postpone calls to `AI.MODELEXECUTE` until the batch's size had reached `m`. In this case, note that requests for which `m` is not reached will hang indefinitely (default value: 0), unless `MINBATCHTIMEOUT` is provided.
* **MINBATCHTIMEOUT**: when provided with a `t` (expressed in milliseconds) that is greater than 0, the engine will trigger a run even though `MINBATCHSIZE` has not been reached after `t` milliseconds from the time a `MODELEXECUTE` (or the enclosing `DAGEXECUTE`) is enqueued. This only applies to cases where both `BATCHSIZE` and `MINBATCHSIZE` are greater than 0.
* **INPUTS**: denotes that one or more names of the model's input nodes are following, applicable only for TensorFlow models (specifying INPUTS for other backends will cause an error)
* **input_count**: a positive number that indicates the number of following input nodes (also applicable only for TensorFlow) 
* **OUTPUTS**: denotes that one or more names of the model's output nodes are following, applicable only for TensorFlow models (specifying OUTPUTS for other backends will cause an error)
* **output_count**: a positive number that indicates the number of following input nodes (also applicable only for TensorFlow)
* **model**: the Protobuf-serialized model. Since Redis supports strings up to 512MB, blobs for very large models need to be chunked, e.g. `BLOB chunk1 chunk2 ...`.

_Return_

A simple 'OK' string or an error.

**Examples**

This example shows to set a model 'mymodel' key using the contents of a local file with [`redis-cli`](https://redis.io/topics/cli). Refer to the [Clients Page](clients.md) for additional client choices that are native to your programming language:

```
$ cat resnet50.pb | redis-cli -x AI.MODELSTORE mymodel TF CPU TAG imagenet:5.0 INPUTS 1 images OUTPUTS 1 output BLOB
OK
```

## AI.MODELSET
_This command is deprecated and will not be available in future versions. consider using AI.MODELSTORE command instead._
The **`AI.MODELSET`** command stores a model as the value of a key. The command's arguments and effect are both exactly the same as `AI.MODELEXECUTE` command, except that `<input_count>` and `<output_count>` arguments should not be specified for TF backend. 

**Redis API**

```
AI.MODELSET <key> <backend> <device>
    [TAG <tag>] [BATCHSIZE <n> [MINBATCHSIZE <m> [MNBATCHTIMEOUT <t>]]]
    [INPUTS <name> ...] [OUTPUTS <name> ...] BLOB <model>
```

## AI.MODELGET
The **`AI.MODELGET`** command returns a model's metadata and blob stored as a key's value.

**Redis API**

```
AI.MODELGET <key> [META] [BLOB]
```

_Arguments_

* **key**: the model's key name
* **META**: will return only the model's meta information on backend, device, tag and batching parameters
* **BLOB**: will return only the model's blob containing the serialized model

_Return_

An array of alternating key-value pairs as follows:

1. **BACKEND**: the backend used by the model as a String
1. **DEVICE**: the device used to execute the model as a String
1. **TAG**: the model's tag as a String
1. **BATCHSIZE**: The maximum size of any batch of incoming requests. If `BATCHSIZE` is equal to 0 each incoming request is served immediately. When `BATCHSIZE` is greater than 0, the engine will batch incoming requests from multiple clients that use the model with input tensors of the same shape.
1. **MINBATCHSIZE**: The minimum size of any batch of incoming requests.
1. **INPUTS**: array reply with one or more names of the model's input nodes (applicable only for TensorFlow models)
1. **OUTPUTS**: array reply with one or more names of the model's output nodes (applicable only for TensorFlow models)
1. **MINBATCHTIMEOUT**: The time in milliseconds for which the engine will wait before executing a request to run the model, when the number of incoming requests is lower than `MINBATCHSIZE`. When `MINBATCHTIMEOUT` is 0, the engine will not run the model before it receives at least `MINBATCHSIZE` requests.
1. **BLOB**: a blob containing the serialized model as a String. If the size of the serialized model exceeds `MODEL_CHUNK_SIZE` (see `AI.CONFIG` command), then an array of chunks is returned. The full serialized model can be obtained by concatenating the chunks.

**Examples**

Assuming that your model is stored under the 'mymodel' key, you can obtain its metadata with:

```
redis> AI.MODELGET mymodel META
 1) "backend"
 2) "TF"
 3) "device"
 4) "CPU"
 5) "tag"
 6) "imagenet:5.0"
 7) "batchsize"
 8) (integer) 0
 9) "minbatchsize"
10) (integer) 0
11) "inputs"
12) 1) "a"
    2) "b"
13) "outputs"
14) 1) "c"
15) "minbatchtimeout"
16) (integer) 0
```

You can also save it to the local file 'model.ext' with [`redis-cli`](https://redis.io/topics/cli) like so:

```
$ redis-cli AI.MODELGET mymodel BLOB > model.ext
```
Note that for the time being, redis-cli adds additional linefeed character to redirected output so that the model blob retrieved with redis-cli will have an additional linefeed character.

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

!!! note "The `AI.MODELDEL` vs. the `DEL` command"
    The `AI.MODELDEL` is equivalent to the [Redis `DEL` command](https://redis.io/commands/del) and should be used in its stead. This ensures compatibility with all deployment options (i.e., stand-alone vs. cluster, OSS vs. Enterprise).


## AI.MODELEXECUTE
The **`AI.MODELEXECUTE`** command runs a model stored as a key's value using its specified backend and device. It accepts one or more input tensors and store output tensors.

The run request is put in a queue and is executed asynchronously by a worker thread. The client that had issued the run request is blocked until the model run is completed. When needed, tensors data is automatically copied to the device prior to execution.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case).

!!! warning "Intermediate memory overhead"
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

**Redis API**

```
AI.MODELEXECUTE <key> INPUTS <input_count> <input> [input ...] OUTPUTS <output_count> <output> [output ...] [TIMEOUT t]
```

_Arguments_

* **key**: the model's key name
* **INPUTS**: denotes the beginning of the input tensors keys' list, followed by the number of inputs and one or more key names
* **input_count**: a positive number that indicates the number of following input keys. 
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by the number of outputs one or more key names
* **output_count**: a positive number that indicates the number of output keys to follow.
* **TIMEOUT**: the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned

_Return_

A simple 'OK' string, a simple `TIMEDOUT` string, or an error.

**Examples**

Assuming that running the model that's stored at 'mymodel' with the tensor 'mytensor' as input outputs two tensors - 'classes' and 'predictions', the following command does that:

```
redis> AI.MODELEXECUTE mymodel INPUTS 1 mytensor OUTPUTS 2 classes predictions
OK
```

## AI.MODELRUN

_This command is deprecated and will not be available in future versions. consider using `AI.MODELEXECUTE` command instead._   

The **`AI.MODELRUN`** command runs a model stored as a key's value using its specified backend and device. It accepts one or more input tensors and store output tensors.

The run request is put in a queue and is executed asynchronously by a worker thread. The client that had issued the run request is blocked until the model run is completed. When needed, tensors data is automatically copied to the device prior to execution.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case).

!!! warning "Intermediate memory overhead"
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

**Redis API**

```
AI.MODELRUN <key> [TIMEOUT t] INPUTS <input> [input ...] OUTPUTS <output> [output ...]
```

_Arguments_

* **key**: the model's key name
* **TIMEOUT**: the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned
* **INPUTS**: denotes the beginning of the input tensors keys' list, followed by one or more key names
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by one or more key names

_Return_

A simple 'OK' string, a simple `TIMEDOUT` string, or an error.

**Examples**

Assuming that running the model that's stored at 'mymodel' with the tensor 'mytensor' as input outputs two tensors - 'classes' and 'predictions', the following command does that:

```
redis> AI.MODELRUN mymodel INPUTS mytensor OUTPUTS classes predictions
OK
```

## AI._MODELSCAN
The **AI._MODELSCAN** command returns all the models in the database. When using Redis open source cluster, the command shall return all the models that are stored in the local shard. 

!!! warning "Experimental API"
    `AI._MODELSCAN` is an EXPERIMENTAL command that may be removed in future versions.

**Redis API**

```
AI._MODELSCAN
```

_Arguments_

None.

_Return_

An array with an entry per model. Each entry is an array with two entries:

1. The model's key name as a String
1. The model's tag as a String

**Examples**

```
redis> > AI._MODELSCAN
1) 1) "mymodel"
   2) imagenet:5.0
```


## AI.SCRIPTSTORE
The **`AI.SCRIPTSTORE`** command stores a [TorchScript](https://pytorch.org/docs/stable/jit.html) as the value of a key.

**Redis API**

```
AI.SCRIPTSTORE <key> <device> [TAG tag] ENTRY_POINTS <entry_points_count> <entry_point> [<entry_point>...] SOURCE "<script>"
```

_Arguments_


* **key**: the script's key name
* **TAG**: an optional string for tagging the script such as a version number or any arbitrary identifier
* **device**: the device that will execute the model can be of:
    * **CPU**: a CPU device
    * **GPU**: a GPU device
    * **GPU:0**, ..., **GPU:n**: a specific GPU device on a multi-GPU system
* **ENTRY_POINTS** A list of function names in the script to be used as entry points upon execution. Each entry point should have the signature of:
`def entry_point(tensors: List[Tensor], keys: List[str], args: List[str])`.
The purpose of each list is as follows:
    * `tensors`: A list holding the input tensors to the function.
    * `keys`: A list of keys that the torch script is about to preform read/write operations on.
    * `args`: A list of additional arguments to the function. If the desired argument is not from type string, it is up to the caller to cast it to the right type, within the script.
* **script**: a string containing [TorchScript](https://pytorch.org/docs/stable/jit.html) source code

_Return_

A simple 'OK' string or an error.

**Examples**

Given the following contents of the file 'addtwo.py':

```python
def addtwo(tensors: List[Tensor], keys: List[str], args: List[str]):
    a = tensors[0]
    b = tensors[1]
    return a + b
```

It can be stored as a RedisAI script using the CPU device with [`redis-cli`](https://redis.io/topics/rediscli) as follows:

```
$ cat addtwo.py | redis-cli -x AI.SCRIPTSTORE myscript CPU TAG myscript:v0.1 ENTRY_POINTS 1 addtwo SOURCE
OK
```

## AI.SCRIPTSET
_This command is deprecated and will not be available in future versions. consider using AI.SCRIPTSTORE command instead._
The **`AI.SCRIPTSET`** command stores a [TorchScript](https://pytorch.org/docs/stable/jit.html) as the value of a key.

**Redis API**

```
AI.SCRIPTSET <key> <device> [TAG tag] SOURCE "<script>"
```

_Arguments_


* **key**: the script's key name
* **TAG**: an optional string for tagging the script such as a version number or any arbitrary identifier
* **device**: the device that will execute the model can be of:
    * **CPU**: a CPU device
    * **GPU**: a GPU device
    * **GPU:0**, ..., **GPU:n**: a specific GPU device on a multi-GPU system
* **script**: a string containing [TorchScript](https://pytorch.org/docs/stable/jit.html) source code

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
$ cat addtwo.py | redis-cli -x AI.SCRIPTSET myscript CPU TAG myscript:v0.1 SOURCE
OK
```

## AI.SCRIPTGET
The **`AI.SCRIPTGET`** command returns the [TorchScript](https://pytorch.org/docs/stable/jit.html) stored as a key's value.

**Redis API**

Get script metadata and source.

```
AI.SCRIPTGET <key> [META] [SOURCE]
```

_Arguments_

* **key**: the script's key name
* **META**: will return only the script's meta information on device, tag and entry points.
* **SOURCE**: will return only the string containing [TorchScript](https://pytorch.org/docs/stable/jit.html) source code

_Return_

An array with alternating entries that represent the following key-value pairs:
!!!!The command returns a list of key-value strings, namely `DEVICE device TAG tag ENTRY_POINTS [entry_point ...] SOURCE source`.

1. **DEVICE**: the script's device as a String
2. **TAG**: the scripts's tag as a String
3. **SOURCE**: the script's source code as a String
4. **ENTRY_POINTS** will return an array containing the script entry point functions

**Examples**

The following shows how to read the script stored at the 'myscript' key:

```
redis> AI.SCRIPTGET myscript
1) "device"
2) CPU
3) "tag"
4) "myscript:v0.1"
5) "source"
6) def addtwo(a, b):
    return a + b
7) "Entry Points"
8) 1) addtwo
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

!!! note "The `AI.SCRIPTDEL` vs. the `DEL` command"
    The `AI.SCRIPTDEL` is equivalent to the [Redis `DEL` command](https://redis.io/commands/del) and should be used in its stead. This ensures compatibility with all deployment options (i.e., stand-alone vs. cluster, OSS vs. Enterprise).


## AI.SCRIPTEXECUTE

The **`AI.SCRIPTEXECUTE`** command runs a script stored as a key's value on its specified device. It receives a list of Redis keys, a list of input tensors and an additional list of arguments to be used in the script.

The run request is put in a queue and is executed asynchronously by a worker thread. The client that had issued the run request is blocked until the script run is completed. When needed, tensors data is automatically copied to the device prior to execution.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case).

!!! warning "Intermediate memory overhead"
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the TORCH backend (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

**Redis API**

```
AI.SCRIPTEXECUTE <key> <function> 
[KEYS <keys_count> <key> [keys...]]
[INPUTS <input_count> <input> [input ...]]
[ARGS <args_count> <arg> [arg...]]
[OUTPUTS <outputs_count> <output> [output ...]]
[TIMEOUT t]
```

_Arguments_

* **key**: the script's key name.
* **function**: the name of the entry point function to run.
* **KEYS**: Denotes the beginning of a list of Redis key names that the script will access to during its execution, for both read and/or write operations.
* **INPUTS**: Denotes the beginning of the input tensors list, followed by its length and one or more input tensors.
* **ARGS**: Denotes the beginning of a list of additional arguments that a user can send to the script. All args are sent as strings, but can be casted to other types supported by torch script, such as `int`, or `float`. 
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by its length and one or more key names.
* **TIMEOUT**: the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned

Note:
Either `KEYS` or `INPUTS` scopes should be provided this command (one or both scopes are acceptable). Those scopes indicate keyspace access and such, the right shard to execute the command at. Redis will verify that all potential key accesses are done to the right shard.

_Return_

A simple 'OK' string, a simple `TIMEDOUT` string, or an error.

**Examples**

The following is an example of running the previously-created 'myscript' on two input tensors:

```
redis> AI.TENSORSET mytensor1{tag} FLOAT 1 VALUES 40
OK
redis> AI.TENSORSET mytensor2{tag} FLOAT 1 VALUES 2
OK
redis> AI.SCRIPTEXECUTE myscript{tag} addtwo INPUTS 2 mytensor1{tag} mytensor2{tag} OUTPUTS 1 result{tag}
OK
redis> AI.TENSORGET result{tag} VALUES
1) FLOAT
2) 1) (integer) 1
3) 1) "42"
```

An example that supports `List[Tensor]` arguments:
```python
def addn(tensors: List[Tensor], keys: List[str], args: List[str]):
    return torch.stack(tensors).sum()
```

```
redis> AI.TENSORSET mytensor1{tag} FLOAT 1 VALUES 40
OK
redis> AI.TENSORSET mytensor2{tag} FLOAT 1 VALUES 1
OK
redis> AI.TENSORSET mytensor3{tag} FLOAT 1 VALUES 1
OK
redis> AI.SCRIPTEXECUTE myscript{tag} addn INPUTS 3 mytensor1{tag} mytensor2{tag} mytensor3{tag} OUTPUTS 1 result{tag}
OK
redis> AI.TENSORGET result{tag} VALUES
1) FLOAT
2) 1) (integer) 1
3) 1) "42"
```

Note: for the time being, as `AI.SCRIPTSET` is still available to use, `AI.SCRIPTEXECUTE` still supports running functions that are part of scripts stored with `AI.SCRIPTSET` or imported from old RDB/AOF files. Meaning calling `AI.SCRIPTEXECUTE` over a function without the dedicated signature of `(tensors: List[Tensor], keys: List[str], args: List[str])` will yield a "best effort" execution to match the deprecated API `AI.SCRIPTRUN` function execution. This will map `INPUTS` tensors only, to their counterpart input arguments in the function, according to the order which they appear.

### Redis Commands support.
In RedisAI TorchScript now supports simple (non-blocking) Redis commands via the `redis.execute` API. The following script gets a key name (`x{1}`), and an `int` value (3). First, the script `SET`s the value in the key. Next, the script `GET`s the value back from the key, and sets it in a tensor which is eventually stored under the key 'y{1}'. Note that the inputs are `str` and `int`. The script sets and gets the value and set it into a tensor.

```
def redis_int_to_tensor(redis_value: int):
    return torch.tensor(redis_value)

def int_set_get(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    value = args[0]
    redis.execute("SET", key, value)
    res = redis.execute("GET", key)
    return redis_string_int_to_tensor(res)
```
```
redis> AI.SCRIPTEXECUTE redis_scripts{1} int_set_get KEYS 1 x{1} ARGS 1 3 OUTPUTS 1 y{1}
OK
redis> AI.TENSORGET y{1} VALUES
1) (integer) 3
```

### RedisAI model execution support.
RedisAI TorchScript also supports executing models which are stored in RedisAI by calling `redisAI.model_execute` command. 
The command receives 3 inputs:
* model name (string)
* model inputs (List of torch.Tensor)
* number of model outputs (int)

* Return value - the model execution output tensors (List of torch.Tensor)

The following script creates two tensors, and executes the (tensorflow) model which is stored under the name 'tf_mul{1}' with these two tensors as inputs.
```
def test_model_execute(tensors: List[Tensor], keys: List[str], args: List[str]):
    a = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    b = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    return redisAI.model_execute(keys[0], [a, b], 1) # assume keys[0] is the model name stored in RedisAI.
```
```
redis> AI.SCRIPTEXECUTE redis_scripts{1} test_model_execute KEYS 1 tf_mul{1} OUTPUTS 1 y{1}
OK
redis> AI.TENSORGET y{1} VALUES
1) (float) 4
2) (float) 9
3) (float) 4
4) (float) 9
```

!!! warning "Intermediate memory overhead"
    The execution of scripts may generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

## AI.SCRIPTRUN
_This command is deprecated and will not be available in future versions. consider using AI.SCRIPTEXECUTE command instead._   

The **`AI.SCRIPTRUN`** command runs a script stored as a key's value on its specified device. It accepts one or more input tensors and store output tensors.

The run request is put in a queue and is executed asynchronously by a worker thread. The client that had issued the run request is blocked until the script run is completed. When needed, tensors data is automatically copied to the device prior to execution.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case).

!!! warning "Intermediate memory overhead"
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the TORCH backend (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

**Redis API**

```
AI.SCRIPTRUN <key> <function> [TIMEOUT t] INPUTS <input> [input ...] [$ input ...] OUTPUTS <output> [output ...]
```

_Arguments_

* **key**: the script's key name
* **function**: the name of the function to run
* **TIMEOUT**: the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned
* **INPUTS**: denotes the beginning of the input tensors keys' list, followed by one or more key names;
              variadic arguments are supported by prepending the list with `$`, in this case the
              script is expected an argument of type `List[Tensor]` as its last argument
* **OUTPUTS**: denotes the beginning of the output tensors keys' list, followed by one or more key names

_Return_

A simple 'OK' string, a simple `TIMEDOUT` string, or an error.

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

If 'myscript' supports variadic arguments:
```python
def addn(a, args : List[Tensor]):
    return a + torch.stack(args).sum()
```

then one can provide an arbitrary number of inputs after the `$` sign:

```
redis> AI.TENSORSET mytensor1 FLOAT 1 VALUES 40
OK
redis> AI.TENSORSET mytensor2 FLOAT 1 VALUES 1
OK
redis> AI.TENSORSET mytensor3 FLOAT 1 VALUES 1
OK
redis> AI.SCRIPTRUN myscript addn INPUTS mytensor1 $ mytensor2 mytensor3 OUTPUTS result
OK
redis> AI.TENSORGET result VALUES
1) FLOAT
2) 1) (integer) 1
3) 1) "42"
```

!!! warning "Intermediate memory overhead"
    The execution of scripts may generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

## AI._SCRIPTSCAN
The **AI._SCRIPTSCAN** command returns all the scripts in the database. When using Redis open source cluster, the command shall return all the scripts that are stored in the local shard. 

!!! warning "Experimental API"
    `AI._SCRIPTSCAN` is an EXPERIMENTAL command that may be removed in future versions.

**Redis API**

```
AI._SCRIPTSCAN
```

_Arguments_

None.

_Return_

An array with an entry per script. Each entry is an array with two entries:

1. The script's key name as a String
1. The script's tag as a String

**Examples**

```
redis> > AI._SCRIPTSCAN
1) 1) "myscript"
   2) "myscript:v0.1"
```

## AI.DAGEXECUTE
The **`AI.DAGEXECUTE`** command specifies a direct acyclic graph of operations to run within RedisAI.

It accepts one or more operations, split by the pipe-forward operator (`|>`).

By default, the DAG execution context is local, meaning that tensor keys appearing in the DAG only live in the scope of the command. That is, setting a tensor with `TENSORSET` will store it in local memory and not set it to an actual database key. One can refer to that key in subsequent commands within the DAG, but that key won't be visible outside the DAG or to other clients - no keys are open at the database level.

Loading and persisting tensors from/to keyspace should be done explicitly. The user should specify which key tensors to load from keyspace using the `LOAD` keyword, and which command outputs to persist to the keyspace using the `PERSIST` keyspace. The user can also specify a tag or key which will assist for the routing of the DAG execution on the right shard in Redis that are going to be accessed for read/write operations (for example, from within `AI.SCRIPTEXECUTE` command), by using the keyword `ROUTING`.  

As an example, if `command 1` sets a tensor, it can be referenced by any further command on the chaining.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case). Note that individual `MODELEXECUTE` or `SCRIPTEXECUTE` commands within the DAG do not support `TIMEOUT`. `TIMEOUT` only applies to the `DAGEXECUTE` request as a whole.


**Redis API**

```
AI.DAGEXECUTE [LOAD <n> <key-1> <key-2> ... <key-n>]
          [PERSIST <n> <key-1> <key-2> ... <key-n>]
          [ROUTING <routing_tag>]
          [TIMEOUT t]
          |> <command> [|>  command ...]
```

_Arguments_

* **LOAD**: denotes the beginning of the input tensors keys' list, followed by the number of keys, and one or more key names
* **PERSIST**: denotes the beginning of the output tensors keys' list, followed by the number of keys, and one or more key names
* **ROUTING**: denotes a key to be used in the DAG or a tag that will assist in routing the dag execution command to the right shard. Redis will verify that all potential key accesses are done to within the target shard.

_While each of the LOAD, PERSIST and ROUTING sections are optional (and may appear at most once in the command), the command must contain **at least one** of these 3 keywords._

* **TIMEOUT**: an optional argument, denotes the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned
* **|> command**: the chaining operator, that denotes the beginning of a RedisAI command, followed by one of RedisAI's commands. Command splitting is done by the presence of another `|>`. The supported commands are:
    * `AI.TENSORSET`
    * `AI.TENSORGET`
    * `AI.MODELEXECUTE`
    * `AI.SCRIPTEXECUTE`


`AI.MODELEXECUTE` and `AI.SCRIPTEXECUTE` commands can run on models or scripts that were set on different devices. RedisAI will analyze the DAG and execute commands in parallel if they are located on different devices and their inputs are available.

_Return_

An array with an entry per command's reply. Each entry format respects the specified command reply.
In case the `DAGEXEUTE` request times out, a `TIMEDOUT` simple string is returned.

**Examples**

Assuming that running the model that's stored at 'mymodel', we define a temporary tensor 'mytensor' and use it as input, and persist only one of the two outputs - discarding 'classes' and persisting 'predictions'. In the same command return the tensor value of 'predictions'.  The following command does that:


```
redis> AI.DAGEXECUTE PERSIST 1 predictions{tag} |>
          AI.TENSORSET mytensor FLOAT 1 2 VALUES 5 10 |>
          AI.MODELEXECUTE mymodel{tag} INPUTS 1 mytensor OUTPUTS 2 classes predictions{tag} |>
          AI.TENSORGET predictions{tag} VALUES
1) OK
2) OK
3) 1) FLOAT
   1) 1) (integer) 2
      1) (integer) 2
   2) "\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
```

A common pattern is enqueuing multiple SCRIPTEXECUTE and MODELEXECUTE commands within a DAG. The following example uses ResNet-50,to classify images into 1000 object categories. Given that our input tensor contains each color represented as a 8-bit integer and that neural networks usually work with floating-point tensors as their input we need to cast a tensor to floating-point and normalize the values of the pixels - for that we will use `pre_process_3ch` function. 

To optimize the classification process we can use a post process script to return only the category position with the maximum classification - for that we will use `post_process` script. Using the DAG capabilities we've removed the necessity of storing the intermediate tensors in the keyspace. You can even run the entire process without storing the output tensor, as follows:

```
redis> AI.DAGEXECUTE ROUTING {tag} |> 
            AI.TENSORSET image UINT8 224 224 3 BLOB b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00....' |> 
            AI.SCRIPTEXECUTE imagenet_script{tag} pre_process_3ch INPUTS 1 image OUTPUTS 1 temp_key1 |> 
            AI.MODELEXECUTE imagenet_model{tag} INPUTS 1 temp_key1 OUTPUTS 1 temp_key2 |> 
            AI.SCRIPTEXECUTE imagenet_script{tag} post_process INPUTS 1 temp_key2 OUTPUTS 1 output |> 
            AI.TENSORGET output VALUES
1) OK
2) OK
3) OK
4) OK
5) 1) 1) (integer) 111
```

As visible on the array reply, the label position with higher classification was 111. 

By combining DAG with multiple SCRIPTEXECUTE and MODELEXECUTE commands we've substantially removed the overall required bandwith and network RX ( we're now returning a tensor with 1000 times less elements per classification ).



!!! warning "Intermediate memory overhead"
    The execution of models and scripts within the DAG may generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

## AI.DAGEXECUTE_RO

The **`AI.DAGEXEUTE_RO`** command is a read-only variant of `AI.DAGEXECUTE`.
`AI.DAGEXECUTE` is flagged as a 'write' command in the Redis command table (as it provides the `PERSIST` option, for example). Hence, read-only cluster replicas will refuse to run the command and it will be redirected to the master even if the connection is using read-only mode.

`AI.DAGEXECUTE_RO` behaves exactly like the original command, excluding the `PERSIST` option and `AI.SCRIPTEXECUTE` commands. It is a read-only command that can safely be with read-only replicas.

!!! info "Further reference"
    Refer to the Redis [`READONLY` command](https://redis.io/commands/readonly) for further information about read-only cluster replicas.

## AI.DAGRUN
_This command is deprecated and will not be available in future versions. consider using `AI.DAGEXECUTE` command instead._
The **`AI.DAGRUN`** command specifies a direct acyclic graph of operations to run within RedisAI.

It accepts one or more operations, split by the pipe-forward operator (`|>`).

By default, the DAG execution context is local, meaning that tensor keys appearing in the DAG only live in the scope of the command. That is, setting a tensor with `TENSORSET` will store it local memory and not set it to an actual database key. One can refer to that key in subsequent commands within the DAG, but that key won't be visible outside the DAG or to other clients - no keys are open at the database level.

Loading and persisting tensors from/to keyspace should be done explicitly. The user should specify which key tensors to load from keyspace using the `LOAD` keyword, and which command outputs to persist to the keyspace using the `PERSIST` keyspace.

As an example, if `command 1` sets a tensor, it can be referenced by any further command on the chaining.

A `TIMEOUT t` argument can be specified to cause a request to be removed from the queue after it sits there `t` milliseconds, meaning that the client won't be interested in the result being computed after that time (`TIMEDOUT` is returned in that case). Note that individual `MODELRUN` or `SCRIPTRUN` commands within the DAG do not support `TIMEOUT`. `TIMEOUT` only applies to the `DAGRUN` request as a whole.


**Redis API**

```
AI.DAGRUN [LOAD <n> <key-1> <key-2> ... <key-n>]
          [PERSIST <n> <key-1> <key-2> ... <key-n>]
          [TIMEOUT t]
          |> <command> [|>  command ...]
```

_Arguments_

* **LOAD**: an optional argument, that denotes the beginning of the input tensors keys' list, followed by the number of keys, and one or more key names
* **PERSIST**: an optional argument, that denotes the beginning of the output tensors keys' list, followed by the number of keys, and one or more key names
* **TIMEOUT**: the time (in ms) after which the client is unblocked and a `TIMEDOUT` string is returned
* **|> command**: the chaining operator, that denotes the beginning of a RedisAI command, followed by one of RedisAI's commands. Command splitting is done by the presence of another `|>`. The supported commands are:
    * `AI.TENSORSET`
    * `AI.TENSORGET`
    * `AI.MODELRUN`
    * `AI.SCRIPTRUN`

`AI.MODELRUN` and `AI.SCRIPTRUN` commands can run on models or scripts that were set on different devices. RedisAI will analyze the DAG and execute commands in parallel if they are located on different devices and their inputs are available.

_Return_

An array with an entry per command's reply. Each entry format respects the specified command reply.
In case the `DAGRUN` request times out, a `TIMEDOUT` simple string is returned.

**Examples**

Assuming that running the model that's stored at 'mymodel', we define a temporary tensor 'mytensor' and use it as input, and persist only one of the two outputs - discarding 'classes' and persisting 'predictions'. In the same command return the tensor value of 'predictions'.  The following command does that:


```
redis> AI.DAGRUN PERSIST 1 predictions |>
          AI.TENSORSET mytensor FLOAT 1 2 VALUES 5 10 |>
          AI.MODELRUN mymodel INPUTS mytensor OUTPUTS classes predictions |>
          AI.TENSORGET predictions VALUES
1) OK
2) OK
3) 1) FLOAT
   2) 1) (integer) 2
      2) (integer) 2
   3) "\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
```

A common pattern is enqueuing multiple SCRIPTRUN and MODELRUN commands within a DAG. The following example uses ResNet-50,to classify images into 1000 object categories. Given that our input tensor contains each color represented as a 8-bit integer and that neural networks usually work with floating-point tensors as their input we need to cast a tensor to floating-point and normalize the values of the pixels - for that we will use `pre_process_3ch` function. 

To optimize the classification process we can use a post process script to return only the category position with the maximum classification - for that we will use `post_process` script. Using the DAG capabilities we've removed the necessity of storing the intermediate tensors in the keyspace. You can even run the entire process without storing the output tensor, as follows:

```
redis> AI.DAGRUN_RO |> 
            AI.TENSORSET image UINT8 224 224 3 BLOB b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00....' |> 
            AI.SCRIPTRUN imagenet_script pre_process_3ch INPUTS image OUTPUTS temp_key1 |> 
            AI.MODELRUN imagenet_model INPUTS temp_key1 OUTPUTS temp_key2 |> 
            AI.SCRIPTRUN imagenet_script post_process INPUTS temp_key2 OUTPUTS output |> 
            AI.TENSORGET output VALUES
1) OK
2) OK
3) OK
4) OK
5) 1) 1) (integer) 111
```

As visible on the array reply, the label position with higher classification was 111. 

By combining DAG with multiple SCRIPTRUN and MODELRUN commands we've substantially removed the overall required bandwith and network RX ( we're now returning a tensor with 1000 times less elements per classification ).



!!! warning "Intermediate memory overhead"
    The execution of models and scripts within the DAG may generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by `maxmemory` configuration settings of Redis.

## AI.DAGRUN_RO
_This command is deprecated and will not be available in future versions. consider using `AI.DAGEXECUTE_RO` command instead._
The **`AI.DAGRUN_RO`** command is a read-only variant of `AI.DAGRUN`.

Because `AI.DAGRUN` provides the `PERSIST` option it is flagged as a 'write' command in the Redis command table. However, even when `PERSIST` isn't used, read-only cluster replicas will refuse to run the command and it will be redirected to the master even if the connection is using read-only mode.

`AI.DAGRUN_RO` behaves exactly like the original command, excluding the `PERSIST` option. It is a read-only command that can safely be with read-only replicas.

!!! info "Further reference"
    Refer to the Redis [`READONLY` command](https://redis.io/commands/readonly) for further information about read-only cluster replicas.

## AI.INFO
The **`AI.INFO`** command returns information about the execution of a model or a script.

Runtime information is collected each time that [`AI.MODELEXECUTE`](#aimodelrun) or [`AI.SCRIPTEXECUTE`](#aiscriptrun) is called. The information is stored locally by the executing RedisAI engine, so when deployed in a cluster each shard stores its own runtime information.

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
 1) key
 2) "myscript"
 3) type
 4) SCRIPT
 5) backend
 6) TORCH
 7) device
 8) CPU
 9) duration
10) (integer) 11391
11) samples
12) (integer) -1
13) calls
14) (integer) 1
15) errors
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
AI.CONFIG <BACKENDSPATH <path>> | <LOADBACKEND <backend> <path>> | <MODEL_CHUNK_SIZE <chunk_size>> | <GET <BACKENDSPATH | MODEL_CHUNK_SIZE>> 
```

_Arguments_

* **BACKENDSPATH**: Specifies the default base backends path to `path`. The backends path is used when dynamically loading a backend (default: '{module_path}/backends', where `module_path` is the module's path).
* **LOADBACKEND**: Loads the DL/ML backend specified by the `backend` identifier from `path`. If `path` is relative, it is resolved by prefixing the `BACKENDSPATH` to it. If `path` is absolute then it is used as is. The `backend` can be one of:
    * **TF**: the TensorFlow backend
    * **TFLITE**: The TensorFlow Lite backend
    * **TORCH**: The PyTorch backend
    * **ONNX**: ONNXRuntime backend
* **MODEL_CHUNK_SIZE**: Sets the size of chunks (in bytes) in which model payloads are split for serialization, replication and `MODELGET`. Default is `511 * 1024 * 1024`.
* **GET**: Retrieve the current value of the `BACKENDSPATH / MODEL_CHUNK_SIZE` configurations. Note that additional information about the module's runtime configuration can be retrieved as part of Redis' info report via `INFO MODULES` command.  

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

This sets model chunk size to one megabyte (not recommended):

```
redis> AI.CONFIG MODEL_CHUNK_SIZE 1048576
OK
```

This returns the current model chunk size configuration:

```
redis> AI.CONFIG GET MODEL_CHUNK_SIZE
1048576
```