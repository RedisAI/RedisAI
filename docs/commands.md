# RedisAI Commands

## AI.TENSORSET

Set a tensor.

Stores a tensor of defined type with shape given by shape1..shapeN.

```sql
AI.TENSORSET tensor_key data_type shape1 shape2 ... [BLOB data | VALUES val1 val2 ...]
```

* tensor_key - Key for storing the tensor
* data_type - Numeric data type of tensor elements, one of FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16
* shape - Shape of the tensor, that is how many elements for each axis

Optional args:

* BLOB data - provide tensor content as a binary buffer
* VALUES val1 val2 - provide tensor content as individual values

> If no BLOB or VALUES are specified, the tensor is allocated but not initialized to any value.

### TENSORSET Example

> Set a 2x2 tensor at `foo`
> 1 2
> 3 4

```sql
AI.TENSORSET foo FLOAT 2 2 VALUES 1 2 3 4
```

!!! warning "Overhead of `AI.TENSORSET` with the optional arg VALUES"
        
    It is possible to set a tensor by specifying each individual value (VALUES ... ) or the entire tensor content as a binary buffer (BLOB ...). You should always try to use the `BLOB` option since it removes the overhead of parsing each individual value and does not require serialization/deserialization of the tensor, thus reducing the overall command latency an improving the maximum attainable performance of the model server.
---

## AI.TENSORGET

Get a tensor.

```sql
AI.TENSORGET tensor_key [META] [BLOB | VALUES]
```

* tensor_key - Key for the tensor
* META - Return tensor meta data (data type and shape)
* BLOB - Return tensor content as a binary buffer
* VALUES - Return tensor content as a list of values

### TENSORGET Example

Get binary data for tensor at `foo` as a buffer.

```sql
AI.TENSORGET foo BLOB

> ...
```

Get tensor values for tensor at `foo` as a list of numbers.

```sql
AI.TENSORGET foo VALUES

>  1) 1.1234
>  2) 2.5135
>  3) 5.5425
>  4) 4.1524
```

Get meta data about tensor at `foo`

```sql
AI.TENSORGET foo META

>  1) "dtype"
>  2) "FLOAT"
>  3) "shape"
>  4) 1) 4
>  4) 2) 5
```

Get meta data as well as binary data for tensor at `foo`

```sql
AI.TENSORGET foo META BLOB

>  1) "dtype"
>  2) "FLOAT"
>  3) "shape"
>  4) 1) 4
>  4) 2) 5
>  5) "blob"
>  6) ...
```

Get binary data for tensor at `foo`. Meta data is also returned.

```sql
AI.TENSORGET foo META VALUES

>  1) "dtype"
>  2) "FLOAT"
>  3) "shape"
>  4) 1) 4
>  4) 2) 5
>  5) "values"
>  6) 1) 1.1234
>  6) 2) 2.5135
>  6) 3) 5.5425
>  6) 4) 4.1524
```

!!! warning "Overhead of `AI.TENSORGET` with the optional arg VALUES"
        
    It is possible to receive a tensor as a list of each individual value (VALUES ... ) or the entire tensor content as a binary buffer (BLOB ...). You should always try to use the `BLOB` option since it removes the overhead of replying each individual value and does not require serialization/deserialization of the tensor, thus reducing the overall command latency an improving the maximum attainable performance of the model server.
---

## AI.MODELSET

Set a model.

```sql
AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]] [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] BLOB model_blob
```

* model_key - Key for storing the model
* backend - The backend corresponding to the model being set. Allowed values: `TF`, `TORCH`, `ONNX`.
* device - Device where the model is loaded and where the computation will run. Allowed values: `CPU`, `GPU`.
* TAG tag - Optional string tagging the model, such as a version number or other identifier
* BATCHSIZE n - Batch incoming requests from multiple clients if they hit the same model and if input tensors have the same
                shape. Upon MODELRUN, the request queue is visited, input tensors from compatible requests are concatenated
                along the 0-th (batch) dimension, up until BATCHSIZE is exceeded. The model is then run for the entire batch,
                results are unpacked back among the individual requests and the respective clients are unblocked.
                If the batch size of the inputs to the first request in the queue exceeds BATCHSIZE, the request is served
                in any case. Default is 0 (no batching).
* MINBATCHSIZE m - Do not execute a MODELRUN until the batch size has reached MINBATCHSIZE. This is primarily used to force
                   batching during testing, but it can also be used under normal operation. In this case, note that requests
                   for which MINBATCHSIZE is not reached will hang indefinitely.
                   Default is 0 (no minimum batch size).
* INPUTS name1 name2 ... - Name of the nodes in the provided graph corresponding to inputs [`TF` backend only]
* OUTPUTS name1 name2 ... - Name of the nodes in the provided graph corresponding to outputs [`TF` backend only]
* BLOB model_blob - Binary buffer containing the model protobuf saved from a supported backend. Since Redis supports strings
                    up to 512MB, blobs for very large models need to be chunked, e.g. `BLOB chunk1 chunk2 ...`.

### MODELSET Example

```sql
AI.MODELSET resnet18 TORCH GPU BLOB < foo.pt
```

```sql
AI.MODELSET resnet18 TF CPU INPUTS in1 OUTPUTS linear4 BLOB < foo.pb
```

```sql
AI.MODELSET mnist_net ONNX CPU TAG mnist:lenet:v0.1 BLOB < mnist.onnx
```

```sql
AI.MODELSET mnist_net ONNX CPU BATCHSIZE 10 BLOB < mnist.onnx
```

```sql
AI.MODELSET resnet18 TF CPU BATCHSIZE 10 MINBATCHSIZE 6 INPUTS in1 OUTPUTS linear4 BLOB < foo.pb
```

## AI.MODELGET

Get model metadata and optionally its binary blob.

```sql
AI.MODELGET model_key [META] [BLOB]
```

* model_key - Key for the model
* META - Return information on backend, device and tag
* BLOB - Return binary blob containing the serialized model

If META is specified, command returns a list of key-value strings, namely `backend <backend> device <device> tag <tag> [blob <blob>]`.

### AI.MODELGET Example

```sql
AI.MODELGET mnist_net META

> 1) "backend"
> 2) "TORCH"
> 3) "device"
> 4) "CPU"
> 5) "tag"
> 6) "mnist"
```

```sql
AI.MODELGET mnist_net BLOB

> ...
```

```sql
AI.MODELGET mnist_net META BLOB

> 1) "backend"
> 2) "TORCH"
> 3) "device"
> 4) "CPU"
> 5) "tag"
> 6) "mnist"
> 7) "blob"
> 8) ...
```

## AI.MODELDEL

Removes a model at a specified key.

```sql
AI.MODELDEL model_key
```

* model_key - Key for the model

Currently, the command is fully equivalent to calling `DEL` on `model_key`.


## AI.MODELRUN

Run a model.

```sql
AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
```

* model_key - Key for the model
* INPUTS input_key1 ... - Keys for tensors to use as inputs
* OUTPUTS output_key2 ... - Keys for storing output tensors

The request is queued and evaded asynchronously from a separate thread. The client blocks until the computation finishes.

If needed, input tensors are copied to the device specified in `AI.MODELSET` before execution.

### MODELRUN Example

```sql
AI.MODELRUN resnet18 INPUTS image12 OUTPUTS label12
```

!!! warning "Intermediate tensors memory overhead when issuing `AI.MODELRUN` and `AI.SCRIPTRUN`"
        
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by maxmemory settings on Redis.
---

## AI._MODELSCAN

NOTE: `_MODELSCAN` is EXPERIMENTAL and might be removed in future versions.

List all models. Returns a list of (key, tag) pairs.

```sql
AI._MODELSCAN
```

### _MODELSCAN Example

```sql
AI.MODELSET model1 TORCH GPU TAG resnet18:v1 < foo.pt
AI.MODELSET model2 TORCH GPU TAG resnet18:v2 < bar.pt

AI._MODELSCAN

>  1) 1) "model1"
>  1) 2) "resnet18:v1"
>  2) 1) "model2"
>  2) 2) "resnet18:v2"
```
---



## AI.SCRIPTSET

Set a script.

```sql
AI.SCRIPTSET script_key device [TAG tag] SOURCE script_source
```

* script_key - Key for storing the script
* device - The device where the script will execute
* TAG tag - Optional string tagging the script, such as a version number or other identifier
* SOURCE script_source - A string containing [TorchScript](https://pytorch.org/docs/stable/jit.html) source code

### SCRIPTSET Example

Given addtwo.txt as:

```python
def addtwo(a, b):
    return a + b
```

```sql
AI.SCRIPTSET addscript GPU SOURCE < addtwo.txt
```

```sql
AI.SCRIPTSET addscript GPU TAG myscript:v0.1 SOURCE < addtwo.txt
```

## AI.SCRIPTGET

Get script metadata and source.

```sql
AI.SCRIPTGET script_key [META] [SOURCE]
```

* script_key - key for the script
* META - Return information on backend, device and tag
* SOURCE - Return string containing the source code for the script

The command returns a list of key-value strings, namely `device <device> tag <tag> [source <source>]`.

### SCRIPTGET Example

```sql
AI.SCRIPTGET addtwo META

> 1) "device"
> 2) "CPU"
> 3) "tag"
> 4) "v1.0"
```

```sql
AI.SCRIPTGET addtwo SOURCE

> ...
```

```sql
AI.SCRIPTGET addtwo META SOURCE

> 3) "device"
> 4) "CPU"
> 5) "tag"
> 6) "v1.0"
> 7) "source"
> 8) ...
```


## AI.SCRIPTDEL

Removes a script at a specified key.

```sql
AI.SCRIPTDEL script_key
```

* script_key - key for the script

Currently, the command is fully equivalent to calling `DEL` on `script_key`.


## AI.SCRIPTRUN

Run a script.

```sql
AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
```

* tensor_key - Key for the script
* fn_name - Name of the function to execute
* INPUTS input_key1 ... - Keys for tensors to use as inputs
* OUTPUTS output_key1 ... - Keys for storing output tensors

If needed, input tensors are copied to the device specified in `AI.SCRIPTSET` before execution.

### SCRIPTRUN Example

```sql
AI.SCRIPTRUN addscript addtwo INPUTS a b OUTPUTS c
```

!!! warning "Intermediate tensors memory overhead when issuing `AI.MODELRUN` and `AI.SCRIPTRUN`"
        
    The execution of models will generate intermediate tensors that are not allocated by the Redis allocator, but by whatever allocator is used in the backends (which may act on main memory or GPU memory, depending on the device), thus not being limited by maxmemory settings on Redis.
---

## AI._SCRIPTSCAN

NOTE: `_SCRIPTSCAN` is EXPERIMENTAL and might be removed in future versions.

List all scripts. Returns a list of (key, tag) pairs.

```sql
AI._SCRIPTSCAN
```

### _SCRIPTSCAN Example

```sql
AI.SCRIPTSET script1 GPU TAG addtwo:v0.1 < addtwo.txt
AI.SCRIPTSET script2 GPU TAG addtwo:v0.2 < addtwo.txt

AI._SCRIPTSCAN

>  1) 1) "script1"
>  1) 2) "addtwo:v0.1"
>  2) 1) "script2"
>  2) 2) "addtwo:v0.2"
```
---

## AI.INFO

Return information about runs of a `MODEL` or a `SCRIPT`.

At each `MODELRUN` or `SCRIPTRUN`, RedisAI will collect statistcs specific for each `MODEL` or `SCRIPT`,
specific for the node (hence nodes in a cluster will have to be queried individually for their info).
The following information is collected:

- `key`: the key being run
- `type`: either `MODEL` or `SCRIPT`
- `backend`: the type of backend (always `TORCH` for `SCRIPT`)
- `device`: the device where the run has been executed
- `duration`: cumulative duration in microseconds
- `samples`: cumulative number of samples obtained from the 0-th (batch) dimension (for `MODEL` only)
- `calls`: number of calls
- `errors`: number of errors generated after the run has been submitted (i.e. excluding errors generated during parsing of the command)

```sql
AI.INFO <model_or_script_key>
```

Statistcs are accumulated until the same command with an extra `RESETSTAT` argument is called. This resets the statistics relative to the model or script.

```sql
AI.INFO <model_or_script_key> RESETSTAT
```

The command can be called on a key until that key is removed using `MODELDEL` or `SCRIPTDEL`.

### AI.INFO Example

```sql
AI.INFO amodel

>  1) key
>  2) "amodel"
>  3) type
>  4) MODEL
>  5) backend
>  6) TORCH
>  7) device
>  8) CPU
>  9) duration
> 10) (integer) 6511
> 11) samples
> 12) (integer) 2
> 13) calls
> 14) (integer) 1
> 15) errors
> 16) (integer) 0
```

```sql
AI.INFO amodel RESETSTAT

> OK

AI.INFO amodel

>  1) key
>  2) "amodel"
>  3) type
>  4) MODEL
>  5) backend
>  6) TORCH
>  7) device
>  8) CPU
>  9) duration
> 10) (integer) 0
> 11) samples
> 12) (integer) 0
> 13) calls
> 14) (integer) 0
> 15) errors
> 16) (integer) 0
```
