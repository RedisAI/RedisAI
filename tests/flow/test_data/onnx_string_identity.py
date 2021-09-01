import onnx
from onnx import helper, TensorProto, ModelProto, defs

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

model_def = helper.make_model(graph_def)
onnx.checker.check_model(model_def)
onnx.save(model_def, "identity_string.onnx")
