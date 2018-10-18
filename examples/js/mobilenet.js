let fs = require('fs');
let Redis = require('ioredis');
let Jimp = require('jimp');

function argmax(arr, start, end) {
  start = start | 0;
  end = end | arr.length;
  let index = 0;
  let value = arr[index];
  for (let i=start; i<end; i++) {
    if (arr[i] > value) {
      value = arr[i];
      index = i;
    }
  }
  return index;
}

function normalize_rgb(buffer) {
  let npixels = buffer.length / 4;
  let out = new Float32Array(npixels * 3);
  for (let i=0; i<npixels; i++) {
    out[3*i]   = buffer[4*i]   / 128 - 1;
    out[3*i+1] = buffer[4*i+1] / 128 - 1;
    out[3*i+2] = buffer[4*i+2] / 128 - 1;
  }
  return out;
}

function buffer_to_float32array(buffer) {
  let out_array = new Float32Array(buffer.length / 4);
  for (let i=0; i<out_array.length; i++) {
    out_array[i] = buffer.readFloatLE(4*i);
  }
  return out_array;
}

async function run(filenames) {

  let json_labels = fs.readFileSync("imagenet_class_index.json");
  let labels = JSON.parse(json_labels);

  let redis = new Redis({ parser: 'javascript' });

  const graph_filename = '../models/mobilenet_v2_1.4_224_frozen.pb';
  const input_var = 'input';
  const output_var = 'MobilenetV2/Predictions/Reshape_1';

  const buffer = fs.readFileSync(graph_filename, {'flag': 'r'});

  console.log("Setting graph");
  redis.call('DL.GRAPH', 'mobilenet', buffer);

  const image_height = 224;
  const image_width = 224;

  for (i in filenames) {

    console.log("Reading image");
    let input_image = await Jimp.read(filenames[i]);

    let image = input_image.cover(image_width, image_height);
    let normalized = normalize_rgb(image.bitmap.data, image.hasAlpha());

    let buffer = Buffer.from(normalized.buffer);

    console.log("Setting input tensor");
    redis.call('DL.TENSOR', 'input_' + i, 'FLOAT', 4, 1, image_width, image_height, 3, 'BLOB', buffer);

    console.log("Running graph");
    redis.call('DL.RUN', 'mobilenet', 1, 'input_' + i, input_var, 'output_' + i, output_var);

    console.log("Getting output tensor");
    let out_data = await redis.callBuffer('DL.DATA', 'output_' + i);
    let out_array = buffer_to_float32array(out_data);

    label = argmax(out_array);

    console.log(filenames[i], labels[label-1]);
  }
}

let filenames = Array.from(process.argv)
filenames.splice(0, 2);

run(filenames);

