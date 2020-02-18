import time
import threading
import numpy as np
from PIL import Image
import os
from redis import StrictRedis
from redisai import Client, Device, Backend
import ml2rt

import argparse

def run( con, index, start, limit):
    res = con.modelrun('yolo', 'in', [f'out-{index}'])
    print('res: {0}'.format(res))
    if index == limit - 1:
        print('Thread finished: ', time.time() - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--minbatch', default=10, type=int)
    parser.add_argument('--threads', default=1000, type=int)
    parser.add_argument('--range', default=125, type=int)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=6379)

    arguments = parser.parse_args()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')

    con = Client(host=arguments.host, port=arguments.port)
    rd = StrictRedis(host=arguments.host, port=arguments.port)
    full_path = os.path.join(test_data_path, 'tinyyolo.pb')
    print("Making sure yolo and in keys are empty")
    rd.delete(['yolo', 'in'])
    print("Reading model from {0}".format(full_path))

    model = ml2rt.load_model(full_path)
    print(f"Setting model with len = {0}".format(len(model)))

    reply = con.modelset(
        'yolo',
        Backend.tf,
        Device.cpu,
        model,
        inputs=['input'],
        outputs=['output'])
    print(reply)
    print("Reading image")
    img_jpg = Image.open(os.path.join(test_data_path, 'sample_dog_416.jpg'))
    # normalize
    img = np.array(img_jpg).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img /= 256.0
    print(f"Setting tensor")
    con.tensorset('in', img)

    limit = arguments.threads
    for _ in range(arguments.range):
        threads = []
        start = time.time()
        for i in range(limit):
            t = threading.Thread(target=run, args=(con, i, start, limit))
            threads.append(t)
            t.start()
            print(f"\nStarting : {i}")
        print("All threads triggered in: ", time.time() - start)
        print("waiting for threads to stop ")
        total_out = 0
        for th in threads:
            th.join()
            total_out = total_out + 1
            print("total ended threads {0}".format(total_out))
    print("purging memory")
    args = ["MEMORY","PURGE"]
    rd.execute_command(*args)
