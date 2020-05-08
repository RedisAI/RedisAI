# RedisAI low-level API

The RedisAI low-level API makes RedisAI available as a library that can be used by other Redis modules written in C or Rust.
Other modules will be able to use this API by calling the function RedisModule_GetSharedAPI() and casting the return value to the right function pointer.

## Public Functions Documentation

### RedisAI\_ClearError


```c
void RedisAI_ClearError (
    RedisAI_Error * err
) 
```


Resets an previously used/allocated **RAI\_Error**



**Parameters:**


* `err` 



        

### RedisAI\_FreeError


```c
void RedisAI_FreeError (
    RedisAI_Error * err
) 
```


Frees the memory of the **RAI\_Error**



**Parameters:**


* `err` 



        

### RedisAI\_GetError


```c
const char * RedisAI_GetError (
    RedisAI_Error * err
) 
```


Return the error description



**Parameters:**


* `err` 



**Returns:**

error description 




**Parameters:**


* `err` 



        

### RedisAI\_GetErrorCode


```c
RedisAI_ErrorCode RedisAI_GetErrorCode (
    RedisAI_Error * err
) 
```


Return the error code



**Parameters:**


* `err` 



**Returns:**

error code 




**Parameters:**


* `err` 



        

### RedisAI\_GetErrorOneLine


```c
const char * RedisAI_GetErrorOneLine (
    RedisAI_Error * err
) 
```


Return the error description as one line



**Parameters:**


* `err` 



**Returns:**

error description as one line 




**Parameters:**


* `err` 



        

### RedisAI\_InitError


```c
int RedisAI_InitError (
    RedisAI_Error ** err
) 
```


Allocate the memory and initialise the **RAI\_Error**.



**Parameters:**


* `result` Output parameter to capture allocated **RAI\_Error**. 



**Returns:**

0 on success, or 1 if the allocation failed. 




        

### RedisAI\_ModelCreate


```c
RedisAI_Model * RedisAI_ModelCreate (
    RedisAI_Backend backend,
    const char * devicestr,
    const char * tag,
    RedisAI_ModelOpts opts,
    size_t ninputs,
    const char ** inputs,
    size_t noutputs,
    const char ** outputs,
    const char * modeldef,
    size_t modellen,
    RedisAI_Error * err
) 
```


Helper method to allocated and initialize a **RAI\_Model**. Depending on the backend it relies on either `model_create_with_nodes` or `model_create` callback functions.



**Parameters:**


* `backend` enum identifying the backend. one of RAI\_BACKEND\_TENSORFLOW, RAI\_BACKEND\_TFLITE, RAI\_BACKEND\_TORCH, or RAI\_BACKEND\_ONNXRUNTIME 
* `devicestr` device string 
* `tag` optional model tag 
* `opts` `RedisAI_ModelOpts` like batchsize or parallelism settings 
* `ninputs` optional number of inputs definition 
* `inputs` optional inputs array 
* `noutputs` optional number of outputs 
* `outputs` optional number of outputs array 
* `modeldef` encoded model definition 
* `modellen` length of the encoded model definition 
* `error` error data structure to store error message in the case of failures 



**Returns:**

**RAI\_Model** model structure on success, or NULL if failed 




        

### RedisAI\_ModelFree


```c
void RedisAI_ModelFree (
    RedisAI_Model * model,
    RedisAI_Error * err
) 
```


Frees the memory of the **RAI\_Model** when the model reference count reaches 0. It is safe to call this function with a NULL input model.



**Parameters:**


* `model` input model to be freed 
* `error` error data structure to store error message in the case of failures 



        

### RedisAI\_ModelGetShallowCopy


```c
RedisAI_Model * RedisAI_ModelGetShallowCopy (
    RedisAI_Model * model
) 
```


Every call to this function, will make the **RAI\_Model** 'model' requiring an additional call to **RAI\_ModelFree()** in order to really free the model. Returns a shallow copy of the model.



**Parameters:**


* `input` model 



**Returns:**

model 




        

### RedisAI\_ModelRun


```c
int RedisAI_ModelRun (
    RedisAI_ModelRunCtx ** mctxs,
    long long n,
    RedisAI_Error * err
) 
```


Given the input array of mctxs, run the associated backend session. If the input array of model context runs is larger than one, then each backend's `model_run` is responsible for concatenating tensors, and run the model in batches with the size of the input array. On success, the tensors corresponding to outputs[0,noutputs-1] are placed in each **RAI\_ModelRunCtx** output tensors array. Relies on each backend's `model_run` definition.



**Parameters:**


* `mctxs` array on input model contexts 
* `n` length of input model contexts array 
* `error` error data structure to store error message in the case of failures 



**Returns:**

REDISMODULE\_OK if the underlying backend `model_run` runned successfully, or REDISMODULE\_ERR if failed. 




        

### RedisAI\_ModelRunCtxAddInput


```c
int RedisAI_ModelRunCtxAddInput (
    RedisAI_ModelRunCtx * mctx,
    const char * inputName,
    RedisAI_Tensor * inputTensor
) 
```


Allocates a **RAI\_ModelCtxParam** data structure, and enforces a shallow copy of the provided input tensor, adding it to the input tensors array of the **RAI\_ModelRunCtx**.



**Parameters:**


* `mctx` input **RAI\_ModelRunCtx** to add the input tensor 
* `inputName` input tensor name 
* `inputTensor` input tensor structure 



**Returns:**

returns 1 on success ( always returns success ) 




        

### RedisAI\_ModelRunCtxAddOutput


```c
int RedisAI_ModelRunCtxAddOutput (
    RedisAI_ModelRunCtx * mctx,
    const char * outputName
) 
```


Allocates a **RAI\_ModelCtxParam** data structure, and sets the tensor reference to NULL ( will be set after MODELRUN ), adding it to the outputs tensors array of the **RAI\_ModelRunCtx**.



**Parameters:**


* `mctx` **RAI\_ModelRunCtx** to add the output tensor 
* `outputName` output tensor name 



**Returns:**

returns 1 on success ( always returns success ) 




        

### RedisAI\_ModelRunCtxCreate


```c
RedisAI_ModelRunCtx * RedisAI_ModelRunCtxCreate (
    RedisAI_Model * model
) 
```


Allocates the **RAI\_ModelRunCtx** data structure required for async background work within `RedisAI_RunInfo` structure on RedisAI blocking commands



**Parameters:**


* `model` input model 



**Returns:**

**RAI\_ModelRunCtx** to be used within 




        

### RedisAI\_ModelRunCtxFree


```c
void RedisAI_ModelRunCtxFree (
    RedisAI_ModelRunCtx * mctx
) 
```


Frees the **RAI\_ModelRunCtx** data structure used within for async background work



**Parameters:**


* `mctx` 



        

### RedisAI\_ModelRunCtxNumOutputs


```c
size_t RedisAI_ModelRunCtxNumOutputs (
    RedisAI_ModelRunCtx * mctx
) 
```


Returns the total number of output tensors of the **RAI\_ModelCtxParam**



**Parameters:**


* `mctx` **RAI\_ModelRunCtx** 



**Returns:**

the total number of output tensors of the **RAI\_ModelCtxParam** 




        

### RedisAI\_ModelRunCtxOutputTensor


```c
RedisAI_Tensor * RedisAI_ModelRunCtxOutputTensor (
    RedisAI_ModelRunCtx * mctx,
    size_t index
) 
```


Get the **RAI\_Tensor** at the output array index position



**Parameters:**


* `mctx` **RAI\_ModelRunCtx** 
* `index` input array index position 



**Returns:**

**RAI\_Tensor** 




        

### RedisAI\_ModelSerialize


```c
int RedisAI_ModelSerialize (
    RedisAI_Model * model,
    char ** buffer,
    size_t * len,
    RedisAI_Error * err
) 
```


Serializes a model given the **RAI\_Model** pointer, saving the serialized data into `buffer` and proving the saved buffer size



**Parameters:**


* `model` **RAI\_Model** pointer 
* `buffer` pointer to the output buffer 
* `len` pointer to the variable to save the output buffer length 
* `error` error data structure to store error message in the case of failures 



**Returns:**

REDISMODULE\_OK if the underlying backend `model_serialize` ran successfully, or REDISMODULE\_ERR if failed. 




        

### RedisAI\_ScriptCreate


```c
RedisAI_Script * RedisAI_ScriptCreate (
    const char * devicestr,
    const char * tag,
    const char * scriptdef,
    RedisAI_Error * err
) 
```


Helper method to allocated and initialize a **RAI\_Script**. Relies on Pytorch backend `script_create` callback function.



**Parameters:**


* `devicestr` device string 
* `tag` script model tag 
* `scriptdef` encoded script definition 
* `error` error data structure to store error message in the case of failures 



**Returns:**

**RAI\_Script** script structure on success, or NULL if failed 




        

### RedisAI\_ScriptFree


```c
void RedisAI_ScriptFree (
    RedisAI_Script * script,
    RedisAI_Error * err
) 
```


Frees the memory of the **RAI\_Script** when the script reference count reaches 0. It is safe to call this function with a NULL input script.



**Parameters:**


* `script` input script to be freed 
* `error` error data structure to store error message in the case of failures 



        

### RedisAI\_ScriptGetShallowCopy


```c
RedisAI_Script * RedisAI_ScriptGetShallowCopy (
    RedisAI_Script * script
) 
```


Every call to this function, will make the **RAI\_Script** 'script' requiring an additional call to **RAI\_ScriptFree()** in order to really free the script. Returns a shallow copy of the script.



**Parameters:**


* `script` input script 



**Returns:**

script 




        

### RedisAI\_ScriptRun


```c
int RedisAI_ScriptRun (
    RedisAI_ScriptRunCtx * sctx,
    RedisAI_Error * err
) 
```


Given the input script context, run associated script session. On success, the tensors corresponding to outputs[0,noutputs-1] are placed in the **RAI\_ScriptRunCtx** output tensors array. Relies on PyTorch's `script_run` definition.



**Parameters:**


* `sctx` input script context 
* `error` error data structure to store error message in the case of failures 



**Returns:**

REDISMODULE\_OK if the underlying backend `script_run` ran successfully, or REDISMODULE\_ERR if failed. 




        

### RedisAI\_ScriptRunCtxAddInput


```c
int RedisAI_ScriptRunCtxAddInput (
    RedisAI_ScriptRunCtx * sctx,
    RedisAI_Tensor * inputTensor
) 
```


Allocates a **RAI\_ScriptCtxParam** data structure, and enforces a shallow copy of the provided input tensor, adding it to the input tensors array of the **RAI\_ScriptRunCtx**.



**Parameters:**


* `sctx` input **RAI\_ScriptRunCtx** to add the input tensor 
* `inputTensor` input tensor structure 



**Returns:**

returns 1 on success ( always returns success ) 




        

### RedisAI\_ScriptRunCtxAddOutput


```c
int RedisAI_ScriptRunCtxAddOutput (
    RedisAI_ScriptRunCtx * sctx
) 
```


Allocates a **RAI\_ScriptCtxParam** data structure, and sets the tensor reference to NULL ( will be set after SCRIPTRUN ), adding it to the outputs tensors array of the **RAI\_ScriptRunCtx**.



**Parameters:**


* `sctx` input **RAI\_ScriptRunCtx** to add the output tensor 



**Returns:**

returns 1 on success ( always returns success ) 




        

### RedisAI\_ScriptRunCtxCreate


```c
RedisAI_ScriptRunCtx * RedisAI_ScriptRunCtxCreate (
    RedisAI_Script * script,
    const char * fnname
) 
```


Allocates the **RAI\_ScriptRunCtx** data structure required for async background work within `RedisAI_RunInfo` structure on RedisAI blocking commands



**Parameters:**


* `script` input script 
* `fnname` function name to used from the script 



**Returns:**

**RAI\_ScriptRunCtx** to be used within 




        

### RedisAI\_ScriptRunCtxFree


```c
void RedisAI_ScriptRunCtxFree (
    RedisAI_ScriptRunCtx * sctx
) 
```


Frees the **RAI\_ScriptRunCtx** data structure used within for async background work



**Parameters:**


* `sctx` 



        

### RedisAI\_ScriptRunCtxNumOutputs


```c
size_t RedisAI_ScriptRunCtxNumOutputs (
    RedisAI_ScriptRunCtx * sctx
) 
```


Returns the total number of output tensors of the **RAI\_ScriptRunCtx**



**Parameters:**


* `sctx` **RAI\_ScriptRunCtx** 



**Returns:**

the total number of output tensors of the **RAI\_ScriptRunCtx** 




        

### RedisAI\_ScriptRunCtxOutputTensor


```c
RedisAI_Tensor * RedisAI_ScriptRunCtxOutputTensor (
    RedisAI_ScriptRunCtx * sctx,
    size_t index
) 
```


Get the **RAI\_Tensor** at the output array index position



**Parameters:**


* `sctx` **RAI\_ScriptRunCtx** 
* `index` input array index position 



**Returns:**

**RAI\_Tensor** 




        

------------------------------
The documentation for this class was generated from the following file `/home/filipe/redislabs/RedisAI/src/script.h`
### RedisAI\_TensorByteSize


```c
size_t RedisAI_TensorByteSize (
    RedisAI_Tensor * t
) 
```


Returns the size in bytes of the underlying deep learning tensor data



**Parameters:**


* `t` input tensor 



**Returns:**

the size in bytes of the underlying deep learning tensor data 




        

### RedisAI\_TensorCreate


```c
RedisAI_Tensor * RedisAI_TensorCreate (
    const char * dataType,
    long long * dims,
    int ndims,
    int hasdata
) 
```


Allocate the memory and initialise the **RAI\_Tensor**. Creates a tensor based on the passed 'dataType`string and with the specified number of dimensions `ndims`, and n-dimension array`dims`.



**Parameters:**


* `dataType` string containing the numeric data type of tensor elements 
* `dims` n-dimensional array ( the dimension values are copied ) 
* `ndims` number of dimensions 
* `hasdata` ( deprecated parameter ) 



**Returns:**

allocated **RAI\_Tensor** on success, or NULL if the allocation failed. 




        

### RedisAI\_TensorCreateByConcatenatingTensors


```c
RedisAI_Tensor * RedisAI_TensorCreateByConcatenatingTensors (
    RedisAI_Tensor ** ts,
    long long n
) 
```


Allocate the memory and initialise the **RAI\_Tensor**, performing a deep copy of the passed array of tensors.



**Parameters:**


* `ts` input array of tensors 
* `n` number of input tensors 



**Returns:**

allocated **RAI\_Tensor** on success, or NULL if the allocation and deep copy failed failed. 




        

### RedisAI\_TensorCreateBySlicingTensor


```c
RedisAI_Tensor * RedisAI_TensorCreateBySlicingTensor (
    RedisAI_Tensor * t,
    long long offset,
    long long len
) 
```


Allocate the memory and initialise the **RAI\_Tensor**, performing a deep copy of the passed tensor, at the given data offset and length.



**Parameters:**


* `t` input tensor 
* `offset` 
* `len` 



**Returns:**

allocated **RAI\_Tensor** on success, or NULL if the allocation and deep copy failed failed. 




        

### RedisAI\_TensorData


```c
char * RedisAI_TensorData (
    RedisAI_Tensor * t
) 
```


Return the pointer the the deep learning tensor data



**Parameters:**


* `t` input tensor 



**Returns:**

direct access to the array data pointer 




        

### RedisAI\_TensorDataSize


```c
size_t RedisAI_TensorDataSize (
    RedisAI_Tensor * t
) 
```


Returns the size in bytes of each element of the tensor



**Parameters:**


* `t` input tensor 



**Returns:**

size in bytes of each the underlying tensor data type 




        

### RedisAI\_TensorDim


```c
long long RedisAI_TensorDim (
    RedisAI_Tensor * t,
    int dim
) 
```


Returns the dimension length for the given input tensor and dimension



**Parameters:**


* `t` input tensor 
* `dim` dimension 



**Returns:**

the dimension length 




        

### RedisAI\_TensorFree


```c
void RedisAI_TensorFree (
    RedisAI_Tensor * t
) 
```


Frees the memory of the **RAI\_Tensor** when the tensor reference count reaches 0. It is safe to call this function with a NULL input tensor.



**Parameters:**


* `t` tensor 



        

### RedisAI\_TensorGetShallowCopy


```c
RedisAI_Tensor * RedisAI_TensorGetShallowCopy (
    RedisAI_Tensor * t
) 
```


Every call to this function, will make the **RAI\_Tensor** 't' requiring an additional call to **RAI\_TensorFree()** in order to really free the tensor. Returns a shallow copy of the tensor.



**Parameters:**


* `t` input tensor 



**Returns:**

shallow copy of the tensor 




        

### RedisAI\_TensorGetValueAsDouble


```c
int RedisAI_TensorGetValueAsDouble (
    RedisAI_Tensor * t,
    long long i,
    double * val
) 
```


Gets the double value from the given input tensor, at the given array data pointer position



**Parameters:**


* `t` tensor to get the data 
* `i` dl\_tensor data pointer position 
* `val` value to set the data to 



**Returns:**

0 on success, or 1 if getting the data failed 




        

### RedisAI\_TensorGetValueAsLongLong


```c
int RedisAI_TensorGetValueAsLongLong (
    RedisAI_Tensor * t,
    long long i,
    long long * val
) 
```


Gets the long value from the given input tensor, at the given array data pointer position



**Parameters:**


* `t` tensor to get the data 
* `i` dl\_tensor data pointer position 
* `val` value to set the data to 



**Returns:**

0 on success, or 1 if getting the data failed 




        

### RedisAI\_TensorNumDims


```c
int RedisAI_TensorNumDims (
    RedisAI_Tensor * t
) 
```


Returns the number of dimensions for the given input tensor



**Parameters:**


* `t` input tensor 



**Returns:**

number of dimensions for the given input tensor 




        

### RedisAI\_TensorSetData


```c
int RedisAI_TensorSetData (
    RedisAI_Tensor * t,
    const char * data,
    size_t len
) 
```


Sets the associated data to the deep learning tensor via deep copying the passed data.



**Parameters:**


* `t` tensor to set the data 
* `data` input data 
* `len` input data length 



**Returns:**

1 on success 




        

### RedisAI\_TensorSetValueFromDouble


```c
int RedisAI_TensorSetValueFromDouble (
    RedisAI_Tensor * t,
    long long i,
    double val
) 
```


Sets the double value for the given tensor, at the given array data pointer position



**Parameters:**


* `t` tensor to set the data 
* `i` dl\_tensor data pointer position 
* `val` value to set the data from 



**Returns:**

0 on success, or 1 if the setting failed 




        

### RedisAI\_TensorSetValueFromLongLong


```c
int RedisAI_TensorSetValueFromLongLong (
    RedisAI_Tensor * t,
    long long i,
    long long val
) 
```


Sets the long value for the given tensor, at the given array data pointer position



**Parameters:**


* `t` tensor to set the data 
* `i` dl\_tensor data pointer position 
* `val` value to set the data from 



**Returns:**

0 on success, or 1 if the setting failed 




        

