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

## AI.MODELSET

Set a model.

```sql
AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
```

* model_key - Key for storing the model
* backend - The backend corresponding to the model being set. Allowed values: `TF`, `TORCH`, `ONNX`.
* device - Device where the model is loaded and where the computation will run. Allowed values: `CPU`, `GPU`.
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
AI.MODELSET mnist_net ONNX CPU < mnist.onnx
```

## AI.MODELGET

Get a model.

```sql
AI.MODELGET model_key
```

* model_key - Key for the model

The command returns the model as serialized by the backend, that is a string containing a protobuf.


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

## AI.SCRIPTSET

Set a script.

```sql
AI.SCRIPTSET script_key device script_source
```

* script_key - Key for storing the script
* device - The device where the script will execute
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

## AI.SCRIPTGET

Get a script.

```sql
AI.SCRIPTGET script_key
```

* script_key - key for the script


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

## AI.CONFIG LOADBACKEND

Enables setting run-time configuration options. See the full documentation about [run time and on load  configurations ](./configuring) for further details.