import json
import os
import random
import sys
import time
from multiprocessing import Process

import numpy as np
from skimage.io import imread
from skimage.transform import resize

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/readies"))
    import paella
except:
    pass

TEST_TF = os.environ.get("TEST_TF") != "0" and os.environ.get("WITH_TF") != "0"
TEST_TFLITE = os.environ.get("TEST_TFLITE") != "0" and os.environ.get("WITH_TFLITE") != "0"
TEST_PT = os.environ.get("TEST_PT") != "0" and os.environ.get("WITH_PT") != "0"
TEST_ONNX = os.environ.get("TEST_ONNX") != "0" and os.environ.get("WITH_ORT") != "0"
DEVICE = os.environ.get('DEVICE', 'CPU').upper()
print(f"Running tests on {DEVICE}\n")


def ensureSlaveSynced(con, env):
    if env.useSlaves:
        # When WAIT returns, all the previous write commands
        # sent in the context of the current connection are
        # guaranteed to be received by the number of replicas returned by WAIT.
        wait_reply = con.execute_command('WAIT', '1', '1000')
        env.assertTrue(wait_reply >= 1)


def check_cuda():
    return os.system('which nvcc')


def info_to_dict(info):
    info = [el.decode('ascii') if type(el) is bytes else el for el in info]
    return dict(zip(info[::2], info[1::2]))


def load_mobilenet_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    labels_filename = os.path.join(test_data_path, 'imagenet_class_index.json')
    image_filename = os.path.join(test_data_path, 'panda.jpg')
    model_filename = os.path.join(test_data_path, 'mobilenet_v2_1.4_224_frozen.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(labels_filename, 'r') as f:
        labels = json.load(f)

    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.float32)

    return model_pb, labels, img


def run_mobilenet(con, img, input_var, output_var):
    time.sleep(0.5 * random.randint(0, 10))
    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELRUN', 'mobilenet',
                        'INPUTS', 'input', 'OUTPUTS', 'output')


def run_test_multiproc(env, n_procs, fn, args=tuple()):
    procs = []

    def tmpfn():
        con = env.getConnection()
        fn(con, *args)
        return 1

    for _ in range(n_procs):
        p = Process(target=tmpfn)
        p.start()
        procs.append(p)

    [p.join() for p in procs]
