import onnx
from onnx import helper, TensorProto, ModelProto, defs


def make_model(graph, **kwargs):  # type: (GraphProto, **Any) -> ModelProto
    model = ModelProto()
    # Touch model.ir_version so it is stored as the version from which it is
    # generated.
    model.ir_version = 7    # this was set manually to suit our (old) onnxruntime version
    model.graph.CopyFrom(graph)

    opset_imports = None  # type: Optional[Sequence[OperatorSetIdProto]]
    opset_imports = kwargs.pop('opset_imports', None)  # type: ignore
    if opset_imports is not None:
        model.opset_import.extend(opset_imports)
    else:
        # Default import
        imp = model.opset_import.add()
        imp.version = 13    # this was set manually to suit our (old) onnxruntime version

    for k, v in kwargs.items():
        # TODO: Does this work with repeated fields?
        setattr(model, k, v)
    return model


# Create string_identity op
string_input = helper.make_tensor_value_info('input:0', TensorProto.STRING, [2, 2])
string_output = helper.make_tensor_value_info('output:0', TensorProto.STRING, [2, 2])
string_identity_def = helper.make_node('Identity',
                                       inputs=['input:0'],
                                       outputs=['output:0'],
                                       name='StringIdentity')

# Make a graph that contains this single op
graph_def = helper.make_graph(nodes=[string_identity_def], name='optional_input_graph', inputs=[string_input],
                              outputs=[string_output])

# Todo: use this call to make the model after upgrading ORT, and remove `make_model` from here
# model_def = helper.make_model(graph_def)
model_def = make_model(graph_def)
onnx.checker.check_model(model_def)
onnx.save(model_def, "identity_string.onnx")
