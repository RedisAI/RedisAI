# From https://github.com/onnx/onnx/issues/2182#issuecomment-513888258

import onnx

def change_input_dim(model):
    sym_batch_dim = -1

    inputs = model.graph.input
    for input in inputs:
        if "Input" not in input.name:
            continue
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = sym_batch_dim

    nodes = model.graph.node
    for node in nodes:
        if node.op_type != 'Reshape':
            continue
        shape_node = [el for el in node.input if 'shape' in el][0]
        shape = [el for el in model.graph.initializer if el.name == shape_node][0]
        if shape.int64_data[0] == 1:
            shape.int64_data[0] = sym_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)


if __name__ == '__main__':
    import sys

    infile = sys.argv[1]
    outfile = sys.argv[2]

    apply(change_input_dim, infile, outfile)

