import json
import os
import random
import sys
import time
from multiprocessing import Process
import threading

import redis
from numpy.random import default_rng
import numpy as np
from skimage.io import imread
from skimage.transform import resize


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../opt/readies"))
import paella

ROOT = os.environ.get("ROOT", None)
TESTMOD_PATH = os.environ.get("TESTMOD", None)
MAX_ITERATIONS = 2 if os.environ.get("MAX_ITERATIONS") == None else os.environ.get("MAX_ITERATIONS")
TEST_TF = os.environ.get("TEST_TF") != "0" and os.environ.get("WITH_TF") != "0"
TEST_TFLITE = os.environ.get("TEST_TFLITE") != "0" and os.environ.get("WITH_TFLITE") != "0"
TEST_PT = os.environ.get("TEST_PT") != "0" and os.environ.get("WITH_PT") != "0"
TEST_ONNX = os.environ.get("TEST_ONNX") != "0" and os.environ.get("WITH_ORT") != "0"
COV = os.environ.get("COV") != "0" and os.environ.get("COV") != "0"
DEVICE = os.environ.get('DEVICE', 'CPU').upper().encode('utf-8', 'ignore').decode('utf-8')
VALGRIND = os.environ.get("VALGRIND") == "1"
print("Running tests on {}\n".format(DEVICE))
print("Using a max of {} iterations per test\n".format(MAX_ITERATIONS))
# change this to make inference tests longer
MAX_TRANSACTIONS=100


def ensureSlaveSynced(con, env, timeout_ms=0):
    if env.useSlaves:
        # When WAIT returns, all the previous write commands
        # sent in the context of the current connection are
        # guaranteed to be received by the number of replicas returned by WAIT.
        wait_reply = con.execute_command('WAIT', '1', timeout_ms)
        try:
            number_replicas = int(wait_reply)
        except Exception as ex:
            # Error in converting to int
            env.debugPring(str(ex), force=True)
            env.assertFalse(True)
            return
        env.assertEqual(number_replicas, 1)


# Ensures command is sent and forced disconnect
# after without waiting for the reply to be parsed
# Usefull for checking behaviour of commands
# that are run with background threads
def send_and_disconnect(cmd, red):
    pool = red.connection_pool
    con = pool.get_connection(cmd[0])
    ret = con.send_command(*cmd)
    con.disconnect()
    # For making sure that Redis will have the time to exit cleanly.
    time.sleep(1)
    return ret


def check_cuda():
    return os.system('which nvcc')


def info_to_dict(info):
    info = [el.decode('utf-8') if type(el) is bytes else el for el in info]
    return dict(zip(info[::2], info[1::2]))


def load_resnet_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data/imagenet')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'dog.jpg')
    model_filename = os.path.join(test_data_path, 'resnet50.pb')
    script_filename = os.path.join(test_data_path, 'data_processing_script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.uint8)

    return model_pb, script, labels, img

def load_resnet_test_data_old():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data/imagenet')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'dog.jpg')
    model_filename = os.path.join(test_data_path, 'resnet50.pb')
    script_filename = os.path.join(test_data_path, 'data_processing_script_old.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.uint8)

    return model_pb, script, labels, img


def load_mobilenet_v1_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'panda.jpg')
    model_filename = os.path.join(test_data_path, 'mobilenet/mobilenet_v1_100_224_cpu_NxHxWxC.pb')
    input_var = 'input'
    output_var = 'MobilenetV1/Predictions/Reshape_1'

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return model_pb, input_var, output_var, labels, img


def load_mobilenet_v2_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'panda.jpg')
    model_filename = os.path.join(test_data_path, 'mobilenet/mobilenet_v2_1.4_224_frozen.pb')
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return model_pb, input_var, output_var, labels, img

def load_creditcardfraud_data(env,max_tensors=10000):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'creditcardfraud.pb')
    creditcard_transaction_filename = os.path.join(test_data_path, 'creditcard_10K.csv')
    rg = default_rng()

    creditcard_transactions = np.genfromtxt(creditcard_transaction_filename, delimiter=',', dtype='float32', skip_header=1, usecols=range(0,30))

    creditcard_referencedata = []
    for tr in range(0,max_tensors):
        creditcard_referencedata.append(rg.random((1,256), dtype='float32'))

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    return model_pb, creditcard_transactions, creditcard_referencedata


def run_mobilenet(con, img, input_var, output_var):
    time.sleep(0.5 * random.randint(0, 10))
    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELEXECUTE', 'mobilenet{1}',
                        'INPUTS', 1, 'input{1}', 'OUTPUTS', 1, 'output{1}')


def run_test_multiproc(env, routing_hint, n_procs, fn, args=tuple()):
    procs = []

    def tmpfn():
        con = env.getConnectionByKey(routing_hint, None)
        fn(con, *args)
        return 1

    for _ in range(n_procs):
        p = Process(target=tmpfn)
        p.start()
        procs.append(p)

    [p.join() for p in procs]


# Load a model/script from a file located in test_data dir.
def load_file_content(file_name):
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    filename = os.path.join(test_data_path, file_name)
    with open(filename, 'rb') as f:
        return f.read()


def check_error_message(env, con, error_msg, *command, error_msg_is_substr=False):
    try:
        con.execute_command(*command)
        env.assertFalse(True)
    except Exception as exception:
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        if error_msg_is_substr:
            # We only verify that the given error_msg is a substring of the entire error message.
            env.assertTrue(str(exception).find(error_msg) >= 0)
        else:
            env.assertEqual(error_msg, str(exception))


def check_error(env, con, *command):
    try:
        con.execute_command(*command)
        env.assertFalse(True)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)


# Returns a dict with all the fields of a certain section from INFO MODULES command
def get_info_section(con, section):
    sections = ['ai_versions', 'ai_git', 'ai_load_time_configs', 'ai_backends_info', 'ai_cpu']
    section_ind = [i for i in range(len(sections)) if sections[i] == 'ai_'+section][0]
    return {k.split(":")[0]: k.split(":")[1]
            for k in con.execute_command("INFO MODULES").decode().split("#")[section_ind+2].split()[1:]}


def get_connection(env, routing_hint):
    return env.getConnectionByKey(routing_hint, None)
