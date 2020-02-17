import time
import threading
import numpy as np
from PIL import Image
import os

from redisai import Client, Device, Backend
import ml2rt

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--minbatch', default=10, type=int)
parser.add_argument('--threads', default=1000, type=int)
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', default=6379)

arguments = parser.parse_args()

test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')

con = Client(host=arguments.host, port=arguments.port)

model = ml2rt.load_model(os.path.join(test_data_path, 'tinyyolo.pb'))

con.modelset(
    'yolo',
    Backend.tf,
    Device.cpu,
    model,
    # batch=arguments.b,
    # minbatch=arguments.m,
    inputs=['input'],
    outputs=['output'])

img_jpg = Image.open(os.path.join(test_data_path, 'sample_dog_416.jpg'))
# normalize
img = np.array(img_jpg).astype(np.float32)
img = np.expand_dims(img, axis=0)
img /= 256.0

con.tensorset('in', img)

limit = arguments.threads


def run(index, start):
    con.modelrun('yolo', 'in', [f'out-{index}'])
    if index == limit - 1:
        print('Thread finished: ', time.time() - start)

for _ in range(125):
    print(f"\nStarting : {_}")
    start = time.time()
    for i in range(limit):
        t = threading.Thread(target=run, args=(i, start))
        t.start()
    print("All threads triggered in: ", time.time() - start)
    time.sleep(5)
