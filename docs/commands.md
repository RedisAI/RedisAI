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
AI.TENSORGET tensor_key [BLOB | VALUES | META]
```

* tensor_key - Key for the tensor
* BLOB - Return tensor content as a binary buffer
* VALUES - Return tensor content as a list of values
* META - Only return tensor meta data (datat type and shape)

### TENSORGET Example

Get binary data for tensor at `foo`. Meta data is also returned.

```sql
AI.TENSORGET foo BLOB
```

!!! warning "Overhead of `AI.TENSORGET` with the optional arg VALUES"
        
    It is possible to receive a tensor as a list of each individual value (VALUES ... ) or the entire tensor content as a binary buffer (BLOB ...). You should always try to use the `BLOB` option since it removes the overhead of replying each individual value and does not require serialization/deserialization of the tensor, thus reducing the overall command latency an improving the maximum attainable performance of the model server.
---

## AI.MODELSET

Set a model.

```sql
AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]] [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
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
* model_blob - Binary buffer containing the model protobuf saved from a supported backend

### MODELSET Example

```sql
AI.MODELSET resnet18 TORCH GPU < foo.pt
```

```sql
AI.MODELSET resnet18 TF CPU INPUTS in1 OUTPUTS linear4 < foo.pb
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

Get model metadata and optionally its binary blob.

```sql
AI.MODELGET model_key [META | BLOB]
```

* model_key - Key for the model
* META - Only return information on backend, device and tag
* BLOB - Return information on backend, device and tag, as well as a binary blob containing the serialized model

The command returns a list of key-value strings, namely `BACKEND backend DEVICE device TAG tag [BLOB blob]`.


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

## AI.MODELLIST

NOTE: `MODELLIST` is EXPERIMENTAL and might be removed in future versions.

List all models. Returns a list of (key, tag) pairs.

```sql
AI.MODELLIST
```

### MODELLIST Example

```sql
AI.MODELSET model1 TORCH GPU TAG resnet18:v1 < foo.pt
AI.MODELSET model2 TORCH GPU TAG resnet18:v2 < bar.pt

AI.MODELLIST

>  1) 1) "model1"
>  1) 2) "resnet18:v1"
>  2) 1) "model2"
>  2) 2) "resnet18:v2"
```
---



## AI.SCRIPTSET

Set a script.

```sql
AI.SCRIPTSET script_key device [TAG tag] script_source
```

* script_key - Key for storing the script
* device - The device where the script will execute
* TAG tag - Optional string tagging the script, such as a version number or other identifier
* script_source - A string containing [TorchScript](https://pytorch.org/docs/stable/jit.html) source code

### SCRIPTSET Example

Given addtwo.txt as:

```python
def addtwo(a, b):
    return a + b
```

```sql
AI.SCRIPTSET addscript GPU < addtwo.txt
```

```sql
AI.SCRIPTSET addscript GPU TAG myscript:v0.1 < addtwo.txt
```

## AI.SCRIPTGET

Get script metadata and source.

```sql
AI.SCRIPTGET script_key
```

* script_key - key for the script

The command returns a list of key-value strings, namely `DEVICE device TAG tag [SOURCE source]`.

## AI.SCRIPTDEL

Removes a script at a specified key.

```sql
AI.SCRIPTDEL script_key
```

* script_key - key for the script

Currently, the command is fully equivalent to calling `DEL` on `script_key`.


### SCRIPTGET Example

```sql
AI.SCRIPTGET addscript
```

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

## AI.SCRIPTLIST

NOTE: `SCRIPTLIST` is EXPERIMENTAL and might be removed in future versions.

List all scripts. Returns a list of (key, tag) pairs.

```sql
AI.SCRIPTLIST
```

### SCRIPTLIST Example

```sql
AI.SCRIPTSET script1 GPU TAG addtwo:v0.1 < addtwo.txt
AI.SCRIPTSET script2 GPU TAG addtwo:v0.2 < addtwo.txt

AI.SCRIPTLIST

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

- `KEY`: the key being run
- `TYPE`: either `MODEL` or `SCRIPT`
- `BACKEND`: the type of backend (always `TORCH` for `SCRIPT`)
- `DEVICE`: the device where the run has been executed
- `DURATION`: cumulative duration in microseconds
- `SAMPLES`: cumulative number of samples obtained from the 0-th (batch) dimension (for `MODEL` only)
- `CALLS`: number of calls
- `ERRORS`: number of errors generated after the run has been submitted (i.e. excluding errors generated during parsing of the command)

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

>  1) KEY
>  2) "amodel"
>  3) TYPE
>  4) MODEL
>  5) BACKEND
>  6) TORCH
>  7) DEVICE
>  8) CPU
>  9) DURATION
> 10) (integer) 6511
> 11) SAMPLES
> 12) (integer) 2
> 13) CALLS
> 14) (integer) 1
> 15) ERRORS
> 16) (integer) 0
```

```sql
AI.INFO amodel RESETSTAT

> OK

AI.INFO amodel

>  1) KEY
>  2) "amodel"
>  3) TYPE
>  4) MODEL
>  5) BACKEND
>  6) TORCH
>  7) DEVICE
>  8) CPU
>  9) DURATION
> 10) (integer) 0
> 11) SAMPLES
> 12) (integer) 0
> 13) CALLS
> 14) (integer) 0
> 15) ERRORS
> 16) (integer) 0
```
