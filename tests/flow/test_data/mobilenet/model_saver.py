import tensorflow as tf
import tensorflow_hub as hub
import ml2rt
import argparse
import sys

url = 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/classification/3'
model_name = 'mobilenet_v1_100_224'
module = hub.Module(url)
batch_size = 1
number_channels = 3
height, width = hub.get_expected_image_size(module)
input_var = 'input'
output_var = 'MobilenetV1/Predictions/Reshape_1'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--input-shape', default="NxHxWxC", type=str)
args = parser.parse_args()
device = 'gpu' if args.gpu else 'cpu'

gpu_available = tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

if gpu_available is False and args.gpu:
    print("No CUDA GPUs found. Exiting...")
    sys.exit(1)

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants

if args.input_shape == "NxHxWxC":
    print("Saving N x H x W x C (1, 224, 224, 3) (with channels_last data format)")
    images = tf.compat.v1.placeholder(tf.float32, shape=(
        batch_size, height, width, number_channels), name=input_var)
elif args.input_shape == "NxHxWxC":
    print("Saving N x C x H x W (1, 3, 224, 224)")
    images = tf.placeholder(tf.float32, shape=(
        batch_size, number_channels, height, width), name=input_var)
else:
    print("inputs shape is either NxHxWxC or NxCxHxW. Exiting...")
    sys.exit(1)

logits = module(images)
logits = tf.identity(logits, output_var)
with tf.compat.v1.Session() as sess:
    sess.run([tf.compat.v1.global_variables_initializer()])
    ml2rt.save_tensorflow(sess, '{model_name}_{device}_{input_shape}.pb'.format(
        model_name=model_name, device=device, input_shape=args.input_shape), output=[output_var])
