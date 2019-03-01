# RedisAI Commands

## AI.TENSORSET - set a tensor
> Stores a tensor of defined type  with shape given by shape1..shapeN

```sql
AI.TENSORSET tensor_key data_type shape1 shape2 ... [BLOB data | VALUES val1 val2 ...]
```

* tensor_key - key for storing the tensor
* data_type - numeric data type of tensor elements, one of FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16
* shape - shape of the tensor, i.e. how many elements for each axis

Optional args:
* BLOB data - provide tensor content as a binary buffer
* VALUES val1 val2 - provide tensor content as individual values

> If no BLOB or VALUES are specified, the tensor is allocated but not initialized to any value.

### Example
> Set a 2x2 tensor at `foo`
> 1 2
> 3 4

```sql
AI.TENSORSET foo FLOAT 2 2 VALUES 1 2 3 4
```

## AI.TENSORGET - get a tensor
```sql
AI.TENSORGET tensor_key [BLOB | VALUES | META]
```

* tensor_key - key for the tensor
* BLOB - return tensor content as a binary buffer
* VALUES - return tensor content as a list of values
* META - only return tensor meta data (datat type and shape)

### Example
> Get binary data for tensor at `foo`. Meta data is also returned.

```sql
AI.TENSORGET foo BLOB
```

## AI.MODELSET - set a model
```sql
AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
```

* model_key - key for storing the model
* backend - the backend corresponding to the model being set. Allowed values: `TF`, `TORCH`.
* device - device where the model is loaded and where the computation will run. Allowed values: `CPU`, `GPU`.
* INPUTS name1 name2 ... - name of the nodes in the provided graph corresponding to inputs [`TF` backend only]
* OUTPUTS name1 name2 ... - name of the nodes in the provided graph corresponding to outputs [`TF` backend only]
* model_blob - binary buffer containing the model protobuf saved from a supported backend

### Example

```sql
AI.MODELSET resnet18 TORCH GPU < foo.pt
```

```sql
AI.MODELSET resnet18 TF CPU INPUTS in1 OUTPUTS linear4 < foo.pt
```

## AI.MODELRUN - run a model
```sql
AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
```

* model_key - key for the model
* INPUTS input_key1 ... - keys for tensors to use as inputs
* OUTPUTS output_key2 ... - keys for storing output tensors

> The request is queued and evaded asynchronously from a separate thread. The client blocks until the computation finishes.

> If needed, input tensors are copied to the device specified in `AI.MODELSET` before execution.

### Example

```sql
AI.MODELRUN resnet18 INPUTS image12 OUTPUTS label12
```


## AI.SCRIPTSET - set a script
```sql
AI.SCRIPTSET script_key device script_source
```

* script_key - key for storing the script
* device - the device where the script will execute
* script_source - a string containing TorchScript source code

### Example

> Given addtwo.txt as:

```python
def addtwo(a, b):
    return a + b
```

```sql
AI.SCRIPTSET addscript GPU < addtwo.txt
```

## AI.SCRIPTGET - get a script

```sql
AI.SCRIPTGET script_key
```

* script_key - key for the script

### Example

```sql
AI.SCRIPTGET addscript
```


## AI.SCRIPTRUN - run a script

```sql
AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
```

* tensor_key - key for the script
* fn_name - name of the function to execute
* INPUTS input_key1 ... - keys for tensors to use as inputs
* OUTPUTS output_key1 ... - keys for storing output tensors

> If needed, input tensors are copied to the device specified in `AI.SCRIPTSET` before execution.


### Example

```sql
AI.SCRIPTRUN addscript addtwo INPUTS a b OUTPUTS c
```
