import redis
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw


labels20 = [
    "aeroplane",     #  0
    "bicycle",       #  1
    "bird",          #  2
    "boat",          #  3
    "bottle",        #  4
    "bus",           #  5
    "car",           #  6
    "cat",           #  7
    "chair",         #  8
    "cow",           #  9
    "diningtable",   # 10
    "dog",           # 11
    "horse",         # 12
    "motorbike",     # 13
    "person",        # 14
    "pottedplant",   # 15
    "sheep",         # 16
    "sofa",          # 17
    "train",         # 18
    "tvmonitor"]     # 19

r = redis.Redis()

with open('tiny-yolo-voc.pb', 'rb') as f:
    blob = f.read()

r.execute_command('AI.MODELSET', 'yolo', 'TF', 'CPU', 'INPUTS', 'input', 'OUTPUTS', 'output', blob)

with open('yolo_boxes.py', 'rb') as f:
    script = f.read()

r.execute_command('AI.SCRIPTSET', 'yolo-post', 'CPU', script)





# filename = "../img/sample_dog_416.jpg"
filename = "../img/sample_office_416.jpg"

img_jpg = Image.open(filename)

# normalize
img = np.array(img_jpg).astype(np.float32)
img -= 128.0
img /= 128.0

r.execute_command('AI.TENSORSET', 'in', 'FLOAT', 1, 416, 416, 3, 'BLOB', img.tobytes())

r.execute_command('AI.MODELRUN', 'yolo', 'INPUTS', 'in', 'OUTPUTS', 'out')

r.execute_command('AI.SCRIPTRUN', 'yolo-post', 'boxes_from_tf', 'INPUTS', 'out', 'OUTPUTS', 'boxes')
typ, shape, buf = r.execute_command('AI.TENSORGET', 'boxes', 'BLOB')
boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)

n_boxes = 0

for box in boxes[0]:

    if box[4] == 0.0:
       continue

    n_boxes += 1

    x1 = img.shape[0] * (box[0] - 0.5 * box[2])
    x2 = img.shape[0] * (box[0] + 0.5 * box[2])
    y1 = img.shape[1] * (box[1] - 0.5 * box[3])
    y2 = img.shape[1] * (box[1] + 0.5 * box[3])

    label = labels20[int(box[-1])]

    draw = ImageDraw.Draw(img_jpg)
    draw.rectangle(((x1, y1), (x2, y2)), outline='green')
    draw.text((x1, y1), label)
    
    img_jpg.save('out.jpg', "JPEG")

print(n_boxes)

