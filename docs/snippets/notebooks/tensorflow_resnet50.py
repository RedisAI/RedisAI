#%%
!pip install --upgrade "redisai>=0.5.0"
!pip install --upgrade "ml2rt"
!pip install --upgrade "scikit-image"
!pip install --upgrade "numpy"
!pip install --upgrade "matplotlib"
!pip install --upgrade "pillow>=7.0.0"

from PIL import Image
import redisai
import json
import redisai as rai
from ml2rt import load_model
import numpy as np
import matplotlib.pyplot as plt

# model, image, and image classes
tf_model_path = './models/tensorflow/imagenet/resnet50.pb'
img_jpg = Image.op/('./data/cat.jpg')
image_classes = json.load(op/("./data/imagenet_classes.json"))

# normalize
raw_image = np.array(img_jpg).astype(np.float32)
img = np.expand_dims(raw_image, axis=0)
img /= 256.0

# load the model from disk
tf_model = load_model(tf_model_path)

# connect to RedisAI
con = redisai.client.Client(host="localhost", port=6379)

# set the model ( only once )
con.modelset(
    'resnet', rai.Backend.tf, device,
    inputs=['images'], outputs=['output'], data=tf_model)

# set the tensor 
tensorset_result = con.tensorset('image', tensor=img)
print(tensorset_result)

# run the model
modelrun_result = con.modelrun('resnet', 'image', 'out')
print(modelrun_result)

# get results
output_tensor = con.tensorget('out', as_numpy=True)

# Get the indices of maximum element in numpy array
result = np.where(output_tensor == np.amax(output_tensor))

print("maximum result {:.2f} belongs to image class at position {} which is {}".format(output_tensor.max(),result[1][0],image_classes[str(result[1][0])]))
# display the image and add image class that it belongs to
plt.figure()
plt.imshow(raw_image)
plt.title(image_classes[str(result[1][0])])
