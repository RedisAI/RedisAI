from RLTest import Env

from multiprocessing import Pool, Process
import redis

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import random
import time
import json
import os
import sys


def run_test_multiproc(env, n_procs, fn, args=tuple()):
    procs = []

    def tmpfn():
        e = env.getConnection()
        fn(e, *args)
        return 1

    for _ in range(n_procs):
        p = Process(target=tmpfn)
        p.start()
        procs.append(p)

    [p.join() for p in procs]


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
    #@@@ this one instead of the above will not blow up, but the test will obviously fail:
    # img = np.zeros([224, 224, 3], dtype=np.float32)

    return model_pb, labels, img


def test_1_run_mobilenet_multiproc(env):
    input_var = 'input'
    output_var = 'MobilenetV2/Predictions/Reshape_1'

    con = env

    model_pb, labels, img = load_mobilenet_test_data()
    con.execute_command('AI.MODELSET', 'mobilenet', 'TF', 'CPU',
                        'INPUTS', input_var, 'OUTPUTS', output_var, model_pb)

    run_test_multiproc(env, 30, run_mobilenet, (img, input_var, output_var))

    dtype, shape, data = con.execute_command('AI.TENSORGET', 'output', 'BLOB')

    dtype_map = {b'FLOAT': np.float32}
    tensor = np.frombuffer(data, dtype=dtype_map[dtype]).reshape(shape)
    label_id = np.argmax(tensor) - 1

    _, label = labels[str(label_id)]

    env.assertEqual(
        label, 'giant_panda'
    )

    #@@@ this one also works as a workaround:
    # env.restartAndReload()


def run_mobilenet(con, img, input_var, output_var):
    time.sleep(0.5 * random.randint(0, 10))
    con.execute_command('AI.TENSORSET', 'input',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    con.execute_command('AI.MODELRUN', 'mobilenet', 'INPUTS', 'input', 'OUTPUTS', 'output')
    # con.execute_command('DEL', 'input')


def test_2_run_mobilenet_multiproc(env):
    test_1_run_mobilenet_multiproc(env)
