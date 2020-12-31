from includes import *
from RLTest import Env

'''
python -m RLTest --test test_serializations.py --module path/to/redisai.so
'''


def tf_model_run(env, model_key):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', model_key, 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

def torch_model_run(env, model_key):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.MODELRUN', model_key, 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

def torch_script_run(env, script_key):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    con.execute_command('AI.SCRIPTRUN', script_key, 'bar', 'INPUTS', 'a{1}', 'b{1}', 'OUTPUTS', 'c{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])


def onnx_model_run(env, model_key):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'features{1}', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)
    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELRUN', model_key, 'INPUTS', 'features{1}', 'OUTPUTS', 'linear_out{1}')
    linear_out = con.execute_command('AI.TENSORGET', 'linear_out{1}', 'VALUES')
    env.assertEqual(float(linear_out[0]), -0.090524077415466309)


class test_v0_rdb_load:

    def __init__(self):
        self.env = Env()

    def test_v0_tf_model(self):
        con = self.env.getConnection()
        key_name = "tf_graph{1}"
        model_rdb = b"\x07\x81\x00\x8f\xff0\xe0\xc4,\x00\x02\x00\x05\x04CPU\x00\x05\x0cTF_GRAPH_V0\x00\x02\x00\x02\x00\x02\x02\x05\x02a\x00\x05\x02b\x00\x02\x01\x05\x02c\x00\x05\xc3@j@\x96\x1f\n,\n\x01a\x12\x0bPlaceholder*\x0b\n\x05dtype\x12\x020\x01*\x05\r\n\x05sha \x0c\x05\x04:\x02\x18\x01\n -\x00b\xe0!-\x15\x19\n\x03mul\x12\x03Mul\x1a\x01a\x1a\x01b*\x07\n\x01T@W\x0f\n\x1b\n\x01c\x12\x08Identity\x1a@'\xe0\x00\x1c\x01\x12\x00\x00\t\x00\xed!\xb2\xef&\xc8\xef."
        con.restore(key_name, 0, model_rdb, True)
        _, backend, _, device, _, tag, _, batchsize, _, minbatchsize, _ , inputs, _, outputs = con.execute_command("AI.MODELGET", key_name, "META")
        self.env.assertEqual([backend, device, tag, batchsize, minbatchsize, inputs, outputs], [b"TF", b"CPU", b"TF_GRAPH_V0", 0, 0, [b'a', b'b'], [b'c']])
        # tf_model_run(self.env, key_name)
    
    def test_v0_torch_model(self):
        key_name = "pt_minimal{1}"
        con = self.env.getConnection()
        model_rdb = b'\x07\x81\x00\x8f\xff0\xe0\xc4,\x00\x02\x02\x05\x04CPU\x00\x05\x0ePT_MINIMAL_V0\x00\x02\x00\x02\x00\x02\x00\x02\x00\x05\xc3C\x0eEH\x0ePK\x03\x04\x00\x00\x08\x08\x00\x00\x86\xb0zO\x00\xe0\x02\x00\x1a\x12\x00\x10\x00pt-minimal/versionFB\x0c\x00Z\xe0\x02\x00\n1\nPK\x07\x08S\xfcQg\x02 ;@\x03\x00P Q\x00\x14 Q\x00\x08\xe0\x08Q\x02\x1c\x004\xe0\x03Q\x13code/__torch__.pyFB0\xe0\x04[\xe0\x1b\x00\x1f5LK\n\x830\x10\xdd{\x8a\xb7T\xb0\x82\xdb\x80\xbd\x81\xbb\xeeJ\t\xa3\x19\xab\x90fd\x12[z\xfb\x1f\x06\xad\xab\xf7\x7f\xb2\xda7k\\$\xd8\xc8\t\x1d\xdab\xf4\x14#\xfao/n\xf3\\\x1eP\x99\x02\xb0v\x1f%\xa5\x17\xa7\xbc\xb06\x97\xef\x8f\xec&\xa5%,\xe1\t\x83A\xc4g\xc7\xf1\x84I\xf4C\xea\xca\xc8~2\x1fy\x99D\xc7\xd9\xda\xe6\xfc\xads\x0f \x83\x1b\x87(z\xc8\xe1\x94\x15.\xd7?5{\xa2\x9c6\r\xd8_\r\x1ar\xae\xa4\x1aC\r\xf2\xebL][\x15?A\x0b\x04a\xc1#I\x8e!\x0b\x00\xc8 \x03\xe1\x11\x0b\x02&\x00\x1e\xe1\x14\x0b\x0c.debug_pklFB\x1a\xe1\x12\x15\x1f5\x8eA\n\xc20\x10EcU\x90\x82+/0\xcb\x8a%\x07p\xe5V\x06t\xdb\x9d\xa4mB"m\xd3\x1f\xa4\x11q\xe7\xca\x1e\xc7S\xa8\xd72)\xe4m\x06\xde\x87?\xff\x99d\xb8)\x88\xc7\x10I\x90\x8cf\x86\xe1\x1f\xbc\xf0]\x9c\xbd\x05\xcf\xc1i[IzU\x8e\x0e\x95U\xbd\xbb\xb4\xdcI]\xa7!\xac\xb9\x00\xa1\xed\x9d\xd9\x1f:\x1bx#r`9\x94\xdb\xfd\x14\x06,w7\xdb\x01\x83\x1d\x94\xa9I\x8a\xb5o\r\x15\xaaS-kh\x15\xff0s\\\x8df\x81G<\xf9\xb7\x1f\x19\x07|et\xbf\xe8\x9cY\xd2a\x08\x04\xa7\x94\x1a\x02\x97!\x04\x00\xb2 \x03A\x08\xe2\rf\x02\x18\x00#\xe1\x05\x08\x07nstants.`\xfa\x00\x1f\xe0\x12\xfa`\x00\x03\x80\x02).Au\x03m/\tW a\x00\x00@\x03\xe0\x11l\x02\x13\x00;\xe0\x03l\x03data\x80g\x007\xe0\x17g\xe0\x0f\x00\x02\x80\x02c\xe2\x00\xc2\x10\nMyModule\nq\x00)\x81}(X#U\x0f\x00trainingq\x01\x88ubq\x02`\xac\x04z\xb8\x18\x811 \x1b@\x03\x02PK\x01C5#0\x83\x82\xe3\x03J\x00\x12 \x17\xe0\x05\x00\xe3\t\x90\x80?\xe3\x01p\xe2\x03~\x00\x1c\xe0\x04<\x00R \r\xe0\x02?\xe3\x08~\xe0\x07I\xe1\x03\xbf\x00& ;\xe0\x01\x00\x01^\x01\xe0\x15I\xe2\x01\xbc\x80S\xe0\x01\xdd\xe1\x03\xa6\x00\x18 \x17\xe0\x01\x00D?\xe0\x04\x9d\xe2\x02\x07\xe0\x07E\xe1\x03?\x00\x13\xe0\x01B \x00\x00\xd4!K\xe0\x02E\xc1\xe0\x04PK\x06\x06, \x1e@\x00\x02\x1e\x03-@\x06`\x00\x00\x05`\x05\xe0\x01\x07\x00e \xd8@\x00\x01\x81\x03@\x05A\x9c\x01\x06\x07 \x06\x01\x00\xe6BV \x00@\x1e\x03PK\x05\x06 \n ;\x00\x05`/@+\x01\x00\x00\x00\t\x00MQ\xab\x8e\xfdc\x97>'
        con.restore(key_name, 0, model_rdb, True)
        _, backend, _, device, _, tag, _, batchsize, _, minbatchsize, _ , inputs, _, outputs = con.execute_command("AI.MODELGET", key_name, "META")
        self.env.assertEqual([backend, device, tag, batchsize, minbatchsize, inputs, outputs], [b"TORCH", b"CPU", b"PT_MINIMAL_V0", 0, 0, [], []])  
        torch_model_run(self.env, key_name)

    def test_v0_troch_script(self):
        key_name = "torch_script{1}"
        con = self.env.getConnection()
        script_rdb = b'\x07\x81\x00\x8f\xd2\t\x12\x0fL\x00\x05\x04CPU\x00\x05\x10TORCH_SCRIPT_V0\x00\x05\xc3@W@i\x0fdef bar(a, b):\n  \x00\x0ereturn a + b\n\nd\x80 \x08_variadic@)\x12args : List[Tensor]\xe0\x06;  \x02[0] A`\t\x031]\n\x00\x00\t\x00\x0b\xee\x04\xe7\x11\xaez\x91'
        con.restore(key_name, 0, script_rdb, True)
        _, device, _, tag = con.execute_command("AI.SCRIPTGET", key_name, "META")
        self.env.assertEqual([device, tag], [b"CPU", b"TORCH_SCRIPT_V0"])  
        torch_script_run(self.env, key_name)

    def test_v0_onnx_model(self):
        key_name = "linear_iris{1}"
        con = self.env.getConnection()
        model_rdb = b'\x07\x81\x00\x8f\xff0\xe0\xc4,\x00\x02\x03\x05\x04CPU\x00\x05\x14ONNX_LINEAR_IRIS_V0\x00\x02\x00\x02\x00\x02\x00\x02\x00\x05\xc3@\xe6A\x15\x17\x08\x05\x12\x08skl2onnx\x1a\x051.4.9"\x07ai.@\x0f\x1f(\x002\x00:\xe2\x01\n\x82\x01\n\x0bfloat_input\x12\x08variabl\x12e\x1a\x0fLinearRegressor"\xe0\x07\x10\x1f*%\n\x0ccoefficients=K\xfe\xc2\xbd=\xf7\xbe\x1c\xbd=/ii>=\x12\xe81\x1a?\xa0\x01\x06*\x14\n\nintercep $\x03\xa8\x1d\xb7= \x15\x01:\n\xa0\x88\x1f.ml\x12 2d76caf265cd4138a74199640a1\x06fc408Z\x1d\xe0\x05\xa5\n\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01 \x03\x03\x04b\x1a\n\xe0\x00\xb7\xe0\x06\x1b\x03\x01B\x0e\n\xe0\x02j\x01\x10\x01\x00\t\x00\x04EU\x04\xd8\\\xdb\x99'
        con.restore(key_name, 0, model_rdb, True)
        _, backend, _, device, _, tag, _, batchsize, _, minbatchsize, _ , inputs, _, outputs = con.execute_command("AI.MODELGET", key_name, "META")
        self.env.assertEqual([backend, device, tag, batchsize, minbatchsize, inputs, outputs], [b"ONNX", b"CPU", b"ONNX_LINEAR_IRIS_V0", 0, 0, [], []])  
        onnx_model_run(self.env, key_name)

    def test_v0_tensor(self):
        key_name = "tensor{1}"
        con = self.env.getConnection()
        tensor_rdb = b'\x07\x81\x00\x8f\xd3\x10\xd4\x8eD\x00\x02\x01\x02\x00\x02 \x02\x00\x02\x01\x02\x02\x02\x02\x02\x01\x02\x01\x02\x01\x02\x00\x05\x08\x01\x00\x00\x00\x02\x00\x00\x00\x00\t\x00^p\x94ty\xf8\xd7y'
        con.restore(key_name, 0, tensor_rdb, True)
        _, tensor_type, _, tensor_shape = con.execute_command('AI.TENSORGET', key_name, 'META')
        self.env.assertEqual([tensor_type, tensor_shape], [b"INT32", [2, 1]])
        values = con.execute_command('AI.TENSORGET', key_name, 'VALUES')
        self.env.assertEqual(values, [1, 2])