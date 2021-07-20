# When update of RDB encoding version is needed, use this script to load models and scripts to RedisAI with the new 
# RDB encoding, dump the content as RDB strings and add those to a the new encoding test on test_serializations.py
import redis

RDB_Version = "V3"


def load_file_content(filename):
    with open(filename, 'rb') as f:
        return f.read()


def main():
    con = redis.Redis()

    # TENSOR
    con.execute_command('AI.TENSORSET', 'tensor{1}', 'INT32', 2, 1, 'VALUES', 1, 2)

    # TF
    model_pb = load_file_content('graph.pb')
    con.execute_command('AI.MODELSTORE', 'tf_graph{1}', 'TF', 'CPU', 
                        'TAG', f'TF_GRAPH_{RDB_Version}', 'BATCHSIZE', 4, 'MINBATCHSIZE', 2, 'MINBATCHTIMEOUT', 1000,
                        'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)

    # TORCH
    model_pb = load_file_content('pt-minimal.pt')
    con.execute_command('AI.MODELSTORE', 'pt_minimal{1}', 'TORCH', 'CPU', 
                        'TAG', f'PT_MINIMAL_{RDB_Version}', 'BATCHSIZE', 4, 'MINBATCHSIZE', 2, 'MINBATCHTIMEOUT', 1000,
                        'BLOB', model_pb)

    # ONNX
    model_pb = load_file_content('linear_iris.onnx')
    con.execute_command('AI.MODELSTORE', 'linear_iris{1}', 'ONNX', 'CPU', 
                        'TAG', f'ONNX_LINEAR_IRIS_{RDB_Version}', 'BATCHSIZE', 4, 'MINBATCHSIZE', 2, 'MINBATCHTIMEOUT', 1000,
                        'BLOB', model_pb)

    # TORCH SCRIPT
    script = load_file_content('script.txt')
    con.execute_command('AI.SCRIPTSTORE', 'torch_script{1}', 'CPU', 
                        'TAG', f'TORCH_SCRIPT_{RDB_Version}', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic',
                        'SOURCE', script)

    print('tensor dump')
    print(con.dump('tensor{1}'))

    print('tf dump')
    print(con.dump('tf_graph{1}'))

    print('torch dump')
    print(con.dump('pt_minimal{1}'))

    print('onnx dump')
    print(con.dump('linear_iris{1}'))

    print('script dump')
    print(con.dump('torch_script{1}'))

if __name__== "__main__":
    main()


