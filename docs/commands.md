# RedisAI Commands

## AI.TENSORSET - Set a tensor
Stores a tensor of defined type (FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16) with N dimensions (dim) and shape given by shape1..shapeN

```sh
AI.TENSORSET tensor_key data_type shape1..shapeN [BLOB data | VALUES val1..valN]
```

## AI.TENSORGET - Get a tensor

```sh
AI.TENSORGET tensor_key [BLOB | VALUES | META]
```

## AI.MODELSET - Set a model
Stores a model provided as a protobuf blob. Backend is TF or TORCH. The TF backend requires the name of input and output nodes to be specified in INPUTS and OUTPUTS.
```sh
AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob
```

## AI.MODELRUN - Run a model
```sh
AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
```

## AI.SCRIPTSET - Set a script
Stores a TorchScript script provided as text.
```sh
AI.SCRIPTSET script_key device script_text
```

## AI.SCRIPTSGET - Get a script
```sh
AI.SCRIPTGET script_key
```

## AI.SCRIPTRUN - Run a script
```sh
AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
```
