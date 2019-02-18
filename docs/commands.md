# RedisAI Commands

## AI.SET TENSOR- Set a tensor
Stores a tensor of defined type (FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16) with N dimensions (dim) and shape given by shape1..shapeN

```sh
AI.SET TENSOR tensor_key data_type dim shape1..shapeN [BLOB data | VALUES val1..valN]
```

## AI.SET GRAPH - Set a model
Stores a graph provided as a protobuf blob. Backend is TF for now.
```sh
AI.SET GRAPH graph_key backend graph_blob prefix
```

## AI.GET TENSOR - Get a tensor
```sh
AI.GET TENSOR tensor_key [BLOB | VALUES | META]
```

## AI.RUN GRAPH - Run a model
```sh
AI.RUN GRAPH graph_key INPUTS ninputs input_key1 ... NAMES input_name_in_graph1 ... OUTPUTS noutputs output_key1 ... NAMES output_name_in_graph1 ...
```
